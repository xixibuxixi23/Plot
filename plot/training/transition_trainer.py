"""M2 losses and metrics; auxiliary HP never updates authoritative state."""
import torch
from torch.nn import functional as F

from plot.kinematics import wrap_angle


def masked_mean(values, mask):
    while mask.ndim<values.ndim:mask=mask.unsqueeze(-1)
    mask=mask.expand_as(values)
    return torch.where(mask,values,torch.zeros_like(values)).sum()/mask.sum().clamp_min(1)


def ordered_event_targets(address, valid, null, queries):
    """Pack time-ordered non-null frame labels into a fixed slot prefix."""
    positive=valid&(address!=null)
    b,t,a=positive.shape;device=address.device
    count=positive.sum(1).clamp_max(queries).long()
    slot_address=torch.zeros((b,a,queries),dtype=torch.long,device=device)
    slot_time=torch.zeros_like(slot_address)
    slot_valid=torch.zeros_like(slot_address,dtype=torch.bool)
    matches=[]
    for bi in range(b):
        for ai in range(a):
            times=positive[bi,:,ai].nonzero(as_tuple=False).flatten()[:queries]
            for qi,ti in enumerate(times.tolist()):
                slot_time[bi,ai,qi]=ti
                slot_address[bi,ai,qi]=address[bi,ti,ai]
                slot_valid[bi,ai,qi]=True
                matches.append((bi,ai,qi,ti))
    return count,slot_address,slot_time,slot_valid,matches


def decode_ordered_events(model, output, null, frames):
    """Scatter each active ordered slot to one frame, resolving collisions once."""
    count=output['event_count_logits'].argmax(-1)
    slot_active=(torch.arange(output['event_time_logits'].shape[2],device=count.device)
                 [None,None]<count[...,None])
    time=output['event_time_logits'].argmax(-1)
    address=output['event_address_logits'].argmax(-1)
    block_logits,_=model.event_payloads(output,address)
    block=block_logits.argmax(-1)
    time_confidence=output['event_time_logits'].float().softmax(-1).amax(-1)
    address_confidence=output['event_address_logits'].float().softmax(-1).amax(-1)
    confidence=time_confidence*address_confidence
    b,a,q=time.shape
    dense_address=address.new_full((b,frames,a),null)
    dense_block=block.new_zeros((b,frames,a))
    dense_confidence=confidence.new_full((b,frames,a),-1.)
    for bi in range(b):
        for ai in range(a):
            for qi in slot_active[bi,ai].nonzero(as_tuple=False).flatten().tolist():
                ti=int(time[bi,ai,qi])
                if confidence[bi,ai,qi]<=dense_confidence[bi,ti,ai]:continue
                dense_confidence[bi,ti,ai]=confidence[bi,ai,qi]
                dense_address[bi,ti,ai]=address[bi,ai,qi]
                dense_block[bi,ti,ai]=block[bi,ai,qi]
    return dense_address,dense_block,count


def matched_edit_counts(pred, pred_block, address, target_block, block_mask, valid,
                        null, tolerance=None):
    """One-to-one address and complete-edit matches, optionally ignoring time."""
    localized=complete=predicted=labels=0
    b,_t,a=pred.shape
    for bi in range(b):
        for ai in range(a):
            prediction_times=((pred[bi,:,ai]!=null)&valid[bi,:,ai]).nonzero(as_tuple=False).flatten().tolist()
            truth_times=block_mask[bi,:,ai].nonzero(as_tuple=False).flatten().tolist()
            labels+=len(truth_times)
            candidates=[]
            for pi,pt in enumerate(prediction_times):
                for gi,gt in enumerate(truth_times):
                    delta=abs(pt-gt)
                    if ((tolerance is None or delta<=tolerance)
                            and pred[bi,pt,ai]==address[bi,gt,ai]):
                        candidates.append((delta,pi,gi,
                            bool(pred_block[bi,pt,ai]==target_block[bi,gt,ai])))
            used_p=set();used_g=set()
            for _delta,pi,gi,payload_correct in sorted(candidates):
                if pi in used_p or gi in used_g:continue
                used_p.add(pi);used_g.add(gi);localized+=1
                complete+=int(payload_correct)
            predicted+=len(prediction_times)
    return localized,complete,predicted,labels


def tolerant_complete_counts(pred, pred_block, address, target_block, block_mask,
                             valid, null, tolerance=1):
    """Backward-compatible complete-edit counts."""
    _,complete,predicted,labels=matched_edit_counts(
        pred,pred_block,address,target_block,block_mask,valid,null,tolerance)
    return complete,predicted,labels


def transition_loss(model, output, inputs, targets, null_weight=.1, objective="full",
                    occurrence_pos_weight=10., occurrence_threshold=.5,
                    event_count_positive_weight=1.):
    if model.cfg.unified_interactions:
        if objective not in {'full','unified'}:
            raise ValueError('unified M2 needs objective=unified/full; no_hp would omit attacks')
        from plot.training.unified_transition import unified_transition_loss
        return unified_transition_loss(model,output,inputs,targets,
                                       event_count_positive_weight=event_count_positive_weight)
    state=targets['state_valid'].bool() & inputs['active'][:,None]
    valid=targets['address_valid'].bool() & state
    address=targets['address']; n=output['address_logits'].shape[-1]
    # Replace invalid addresses before CE: they may point to masked unknown cells.
    safe=torch.where(valid,address,n-1)
    ce=F.cross_entropy(output['address_logits'].reshape(-1,n),safe.reshape(-1),reduction='none').reshape_as(safe)
    weights=torch.where(safe==n-1,null_weight,1.)
    address_loss=masked_mean(ce*weights,valid)
    block,damage=model.payloads(output,safe)
    block_mask=targets['block_valid'].bool() & valid & (safe<2197)
    damage_mask=targets['damage_valid'].bool() & valid & (safe>=2197) & (safe<n-1)
    terms={
        'position':masked_mean(F.smooth_l1_loss(output['pose'][...,:3],targets['pose'][...,:3],reduction='none'),state),
        'angle':masked_mean(1-torch.cos(output['pose'][...,3:]-targets['pose'][...,3:]),state),
        'address':address_loss,
        'block':masked_mean(F.cross_entropy(block.flatten(0,2),targets['block'].flatten(),reduction='none').reshape_as(safe),block_mask),
        'damage':masked_mean((damage-targets['damage']).abs(),damage_mask),
        'hp_aux':masked_mean((output['hp_aux']-targets['hp']).abs()/20.,state),
        'camera_position':masked_mean(F.smooth_l1_loss(output['camera_relative'],targets['camera_relative'],reduction='none'),state),
        'camera_direction':masked_mean(1-F.cosine_similarity(output['camera_direction'],targets['camera_direction'],dim=-1),state),
    }
    if objective not in {"full", "edit", "no_hp", "player"}:raise ValueError("unknown objective")
    if objective in {"no_hp", "player"}:
        held_mask=state&targets.get('held_item_valid',torch.ones_like(state)).bool()
        terms["held_item"]=masked_mean(F.cross_entropy(
            output["held_logits"].flatten(0,2),targets["held_item"].flatten(),
            reduction="none").reshape_as(safe),held_mask)
    if objective=="no_hp":
        positive=valid & (address!=n-1)
        occurrence=F.binary_cross_entropy_with_logits(
            output['occurrence_logits'],positive.float(),reduction='none',
            pos_weight=output['occurrence_logits'].new_tensor(occurrence_pos_weight))
        terms['occurrence']=masked_mean(occurrence,valid)
        local_safe=address.clamp_max(n-2)
        local_ce=F.cross_entropy(output['address_logits'][...,:-1].reshape(-1,n-1),
                                 local_safe.reshape(-1),reduction='none').reshape_as(address)
        terms['address']=masked_mean(local_ce,positive)
        if 'event_count_logits' in output:
            queries=output['event_count_logits'].shape[-1]-1
            count_target,slot_address,slot_time,slot_valid,matches=ordered_event_targets(
                address,valid,n-1,queries)
            active=inputs['active'].bool()
            count_ce=F.cross_entropy(
                output['event_count_logits'].flatten(0,1),count_target.flatten(),
                reduction='none').reshape_as(count_target)
            count_ce=count_ce*torch.where(
                count_target>0,count_ce.new_tensor(event_count_positive_weight),
                count_ce.new_tensor(1.))
            count_ce=masked_mean(count_ce,active)
            terms['event_count']=count_ce
            teacher_block,_=model.event_payloads(output,slot_address)
            graph_zero=0.*(
                output['event_time_logits'].float().sum()
                +output['event_operation_logits'].float().sum()
                +output['event_address_logits'].float().clamp(-1e4,1e4).sum()
                +teacher_block.float().sum())
            if matches:
                bi=torch.tensor([x[0] for x in matches],device=address.device)
                ai=torch.tensor([x[1] for x in matches],device=address.device)
                qi=torch.tensor([x[2] for x in matches],device=address.device)
                ti=torch.tensor([x[3] for x in matches],device=address.device)
                time_logits=output['event_time_logits'][bi,ai,qi].float()
                time_target=torch.zeros_like(time_logits)
                time_target.scatter_(1,ti[:,None],1.)
                for offset in (-1,1):
                    neighbor=ti+offset;inside=(neighbor>=0)&(neighbor<time_logits.shape[-1])
                    time_target[inside,ti[inside]]-=.1
                    time_target[inside,neighbor[inside]]+=.1
                terms['event_time']=-(time_target*F.log_softmax(time_logits,-1)).sum(-1).mean()
                terms['event_address']=F.cross_entropy(
                    output['event_address_logits'][bi,ai,qi].float(),slot_address[bi,ai,qi])
                payload_valid=targets['block_valid'][bi,ti,ai].bool()&(slot_address[bi,ai,qi]<2197)
                terms['event_block']=(F.cross_entropy(
                    teacher_block[bi[payload_valid],ai[payload_valid],qi[payload_valid]].float(),
                    targets['block'][bi[payload_valid],ti[payload_valid],ai[payload_valid]])
                    if payload_valid.any() else graph_zero)
                if 'edit_kind' in targets:
                    kind=targets['edit_kind'][bi,ti,ai].long();kind_valid=(kind==1)|(kind==2)
                    terms['event_operation']=(F.cross_entropy(
                        output['event_operation_logits'][bi[kind_valid],ai[kind_valid],qi[kind_valid]].float(),
                        kind[kind_valid]-1) if kind_valid.any() else graph_zero)
                else:terms['event_operation']=graph_zero
            else:
                terms['event_time']=terms['event_address']=terms['event_block']=terms['event_operation']=graph_zero
        if 'edit_kind' in targets:
            edit_kind=targets['edit_kind'].long()
            kind_ce=F.cross_entropy(output['edit_kind_logits'].flatten(0,2),edit_kind.flatten(),
                                    weight=output['edit_kind_logits'].new_tensor([.1,1.,1.]),
                                    reduction='none').reshape_as(edit_kind)
            edit_active=state & (edit_kind>0)
            progress_mask=state & targets['edit_progress_valid'].bool() & (edit_kind==1)
            process_target_mask=state & targets['edit_target_valid'].bool() & edit_active
            process_address=targets['edit_target'].long().clamp(0,2196)
            process_ce=F.cross_entropy(output['address_logits'][...,:2197].reshape(-1,2197),
                                       process_address.reshape(-1),reduction='none').reshape_as(edit_kind)
            terms['edit_kind']=masked_mean(kind_ce,state)
            terms['edit_age']=masked_mean(F.smooth_l1_loss(
                output['edit_age'],torch.log1p(targets['edit_age_frames'].float()),reduction='none'),edit_active)
            terms['edit_progress']=masked_mean(F.smooth_l1_loss(
                output['edit_progress'],targets['edit_progress'].float(),reduction='none'),progress_mask)
            terms['edit_target']=masked_mean(process_ce,process_target_mask)
    loss=terms["address"]+terms["block"] if objective=="edit" else sum(terms.values())
    if objective=="player":
        loss=sum(terms[k] for k in (
            'position','angle','camera_position','camera_direction','held_item'))
    if objective=="no_hp":
        auxiliary_weights={'edit_kind':.2,'edit_age':.05,'edit_progress':.2,'edit_target':.2}
        excluded={"damage","hp_aux"}
        if 'event_count_logits' in output:
            # Ordered slots replace the independent per-frame occurrence,
            # address and payload objectives as the authoritative event model.
            excluded|={'occurrence','address','block'}
        loss=sum(v*auxiliary_weights.get(k,1.) for k,v in terms.items() if k not in excluded)
    with torch.no_grad():
        if objective=='no_hp' and 'event_count_logits' in output:
            pred,predicted_block,count_pred=decode_ordered_events(model,output,n-1,address.shape[1])
        elif objective=='no_hp':
            nonnull=output['address_logits'][...,:-1].argmax(-1)
            pred=torch.where(output['occurrence_logits'].sigmoid()>=occurrence_threshold,nonnull,n-1)
            predicted_block,_=model.payloads(output,pred);predicted_block=predicted_block.argmax(-1)
        else:pred=output['address_logits'].argmax(-1)
        positive=valid & (address!=n-1); predicted=valid & (pred!=n-1)
        correct=positive & (pred==address)
        detected=positive & predicted
        nonnull=output["address_logits"][...,:-1].argmax(-1)
        localized=positive & (nonnull==address)
        block_correct=(predicted_block==targets["block"] if objective=='no_hp'
                       else block.argmax(-1)==targets["block"])
        if objective!='no_hp':
            predicted_block,_=model.payloads(output,pred);predicted_block=predicted_block.argmax(-1)
        joint=block_mask & (pred==address) & (predicted_block==targets["block"])
        metrics={k:float(v.detach()) for k,v in terms.items()}
        metrics.update(detected_events=int(detected.sum()),
                       localized_events=int(localized.sum()),
                       occurrence_recall=float(detected.sum()/positive.sum().clamp_min(1)),
                       occurrence_precision=float(detected.sum()/predicted.sum().clamp_min(1)),
                       localization_given_event=float(localized.sum()/positive.sum().clamp_min(1)))
        metrics.update(block_labels=int(block_mask.sum()),
                       correct_block_payloads=int((block_correct & block_mask).sum()),
                       correct_edits=int(joint.sum()),
                       block_payload_accuracy=float((block_correct & block_mask).sum()/block_mask.sum().clamp_min(1)),
                       edit_recall=float(joint.sum()/block_mask.sum().clamp_min(1)),
                       edit_precision_lower_bound=float(joint.sum()/predicted.sum().clamp_min(1)),
                       held_accuracy=float(masked_mean((output['held_logits'].argmax(-1)==targets['held_item']).float(),state)),
                       camera_distance=float(masked_mean((output['camera_relative']-targets['camera_relative']).norm(dim=-1),state)),
                       camera_angle_deg=float(masked_mean(torch.acos(F.cosine_similarity(output['camera_direction'].float(),targets['camera_direction'].float(),dim=-1).clamp(-1,1)),state)*180/torch.pi))
        metrics.update(loss=float(loss.detach()),
                       address_accuracy=float(masked_mean((pred==address).float(),valid)),
                       event_recall=float(correct.sum()/positive.sum().clamp_min(1)),
                       event_precision=float(correct.sum()/predicted.sum().clamp_min(1)),
                       event_labels=int(positive.sum()),
                       valid_queries=int(valid.sum()),correct_addresses=int(((pred==address)&valid).sum()),
                       predicted_events=int(predicted.sum()),correct_events=int(correct.sum()),
                       false_events_without_edit_input=int((predicted & ~positive &
                           ~((inputs['actions'][...,8]>0.5)|(inputs['actions'][...,9]>0.5))).sum()),
                       position_error=float(masked_mean((output['pose'][...,:3]-targets['pose'][...,:3]).norm(dim=-1),state)),
                       angle_error_deg=float(masked_mean(wrap_angle(output['pose'][...,3:]-targets['pose'][...,3:]).abs(),state)*180/torch.pi))
        if objective=='no_hp' and 'event_count_logits' in output:
            tolerant_correct,tolerant_predicted,tolerant_labels=tolerant_complete_counts(
                pred,predicted_block,address,targets['block'],block_mask,valid,n-1,tolerance=1)
            tolerant2_localized,tolerant2_correct,_,_=matched_edit_counts(
                pred,predicted_block,address,targets['block'],block_mask,valid,n-1,tolerance=2)
            coordinate_localized,coordinate_correct,_,_=matched_edit_counts(
                pred,predicted_block,address,targets['block'],block_mask,valid,n-1,tolerance=None)
            queries=output['event_time_logits'].shape[2]
            target_count=positive.sum(1).clamp_max(queries)
            active=inputs['active'].bool()
            metrics.update(
                tolerant_correct_edits=tolerant_correct,
                tolerant_predicted_events=tolerant_predicted,
                tolerant_block_labels=tolerant_labels,
                tolerant2_localized_edits=tolerant2_localized,
                tolerant2_correct_edits=tolerant2_correct,
                coordinate_localized_edits=coordinate_localized,
                coordinate_correct_edits=coordinate_correct,
                false_coordinate_writes=tolerant_predicted-coordinate_localized,
                false_complete_writes=tolerant_predicted-coordinate_correct,
                payload_correct_given_coordinate=coordinate_correct/max(1,coordinate_localized),
                edit_precision_tolerance1=tolerant_correct/max(1,tolerant_predicted),
                edit_recall_tolerance1=tolerant_correct/max(1,tolerant_labels),
                event_count_accuracy=float(masked_mean((count_pred==target_count).float(),active)),
                event_count_mae=float(masked_mean((count_pred-target_count).abs().float(),active)))
            tp=metrics['edit_precision_tolerance1'];tr=metrics['edit_recall_tolerance1']
            metrics['edit_f1_tolerance1']=2*tp*tr/max(1e-12,tp+tr)
    return loss,metrics
