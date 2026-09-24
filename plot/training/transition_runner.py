"""Shared M2 batch, evaluation, and diagnostic visualization helpers."""

import numpy as np
import torch

from plot.training.transition_trainer import transition_loss


def move(batch,device):
    return {g:{k:v.to(device) for k,v in batch[g].items()} for g in ('inputs','targets')}


@torch.no_grad()
def evaluate(model,loader,device,max_batches=4,objective="full",null_weight=.1,
             occurrence_pos_weight=10.,occurrence_threshold=.5):
    model.eval();metrics=[];example=None;best_score=-1
    for index,batch in enumerate(loader):
        if max_batches is not None and max_batches>0 and index>=max_batches:break
        values=move(batch,device);out=model(values['inputs'])
        _,m=transition_loss(model,out,**values,objective=objective,null_weight=null_weight,
                           occurrence_pos_weight=occurrence_pos_weight,
                           occurrence_threshold=occurrence_threshold);metrics.append(m)
        score=int(((values['targets']['damage_valid']|values['targets']['block_valid'])&values['targets']['state_valid']).sum())
        if score>best_score:example=(batch,values,out);best_score=score
    model.train()
    result={k:float(np.mean([row[k] for row in metrics])) for k in metrics[0]}
    count_keys=('event_labels','valid_queries','correct_addresses','predicted_events','correct_events',
                'block_labels','correct_block_payloads','correct_edits','detected_events','localized_events',
                'false_events_without_edit_input','tolerant_correct_edits','tolerant_predicted_events',
                'tolerant_block_labels','tolerant2_localized_edits','tolerant2_correct_edits',
                'coordinate_localized_edits','coordinate_correct_edits','false_coordinate_writes',
                'false_complete_writes')
    for k in count_keys:
        if k in metrics[0]:result[k]=sum(row[k] for row in metrics)
    result['event_recall']=result['correct_events']/max(1,result['event_labels'])
    result['event_precision']=result['correct_events']/max(1,result['predicted_events'])
    result['address_accuracy']=result['correct_addresses']/max(1,result['valid_queries'])
    result['block_payload_accuracy']=result['correct_block_payloads']/max(1,result['block_labels'])
    result['edit_recall']=result['correct_edits']/max(1,result['block_labels'])
    result['occurrence_recall']=result['detected_events']/max(1,result['event_labels'])
    result['occurrence_precision']=result['detected_events']/max(1,result['predicted_events'])
    result['localization_given_event']=result['localized_events']/max(1,result['event_labels'])
    result['edit_precision_lower_bound']=result['correct_edits']/max(1,result['predicted_events'])
    if 'tolerant2_correct_edits' in result:
        predicted=result['tolerant_predicted_events'];labels=result['tolerant_block_labels']
        result['edit_precision_tolerance2']=result['tolerant2_correct_edits']/max(1,predicted)
        result['edit_recall_tolerance2']=result['tolerant2_correct_edits']/max(1,labels)
        result['coordinate_precision']=result['coordinate_localized_edits']/max(1,predicted)
        result['coordinate_recall']=result['coordinate_localized_edits']/max(1,labels)
        result['coordinate_payload_precision']=result['coordinate_correct_edits']/max(1,predicted)
        result['coordinate_payload_recall']=result['coordinate_correct_edits']/max(1,labels)
        result['payload_correct_given_coordinate']=(
            result['coordinate_correct_edits']/max(1,result['coordinate_localized_edits']))
    p,r=result['edit_precision_lower_bound'],result['edit_recall']
    result['edit_f1_lower_bound']=2*p*r/max(1e-12,p+r)
    return result,example


def visualize(example,path,step,objective="full"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    batch,values,out=example
    truth=values['targets'];inputs=values['inputs']
    index=int(((truth['damage_valid']|truth['block_valid'])&truth['state_valid']).sum((1,2)).argmax())
    a=int(inputs['active'][index].sum())
    pred=out['pose'][index,:,:a].detach().cpu().numpy();gt=truth['pose'][index,:,:a].cpu().numpy()
    hp=truth['hp'][index,:,:a].cpu().numpy();aux=out['hp_aux'][index,:,:a].detach().cpu().numpy()
    addresses=out['address_logits'][index,:,:a].argmax(-1).cpu().numpy();labels=truth['address'][index,:,:a].cpu().numpy()
    valid=truth['address_valid'][index,:,:a].cpu().numpy()
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for i in range(a):
        line=axes[0].plot(gt[:,i,0],gt[:,i,1],'-o',label=f'agent{i} true')[0]
        axes[0].plot(pred[:,i,0],pred[:,i,1],'--x',color=line.get_color(),label=f'agent{i} model')
        axes[1].plot(range(1,9),hp[:,i],color=line.get_color(),label=f'agent{i} true')
        axes[1].plot(range(1,9),aux[:,i],'--',color=line.get_color(),label=f'agent{i} auxiliary')
    axes[0].set_title('XY trajectory');axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(fontsize=6)
    axes[1].set_title('HP (auxiliary prediction, not committed)');axes[1].legend(fontsize=6)
    if objective=='edit':
        axes[0].clear();axes[1].clear()
        for i in range(a):
            good=valid[:,i] & (labels[:,i]<2197)
            t=np.arange(1,9)[good]
            axes[0].scatter(t,labels[good,i],marker='o',label=f'agent{i} true')
            axes[0].scatter(t,addresses[good,i],marker='x',label=f'agent{i} model')
        axes[0].set_aspect('auto');axes[0].set_title('Address at true edits (null >=2197+A)');axes[0].legend(fontsize=6)
        axes[0].set_xlabel('Transition');axes[0].set_ylabel('Local voxel index')
        axes[1].imshow(inputs['previous_rgb'][index,0].detach().cpu().permute(1,2,0).numpy())
        axes[1].set_title('Initial RGB condition: agent0');axes[1].axis('off')
    # 0 voxel, 1 character, 2 null; rows alternate truth and model for each resident.
    null=out['address_logits'].shape[-1]-1
    kinds=lambda x:np.where(x<2197,0,np.where(x<null,1,2))
    rows=np.empty((a*2,8));rows[0::2]=np.where(valid,kinds(labels),np.nan).T;rows[1::2]=kinds(addresses).T
    axes[2].imshow(rows,vmin=0,vmax=2,aspect='auto',interpolation='nearest')
    axes[2].set_yticks(range(a*2),[f'agent{i} {s}' for i in range(a) for s in ('true','model')])
    axes[2].set_title('Write kind: voxel=0 / resident=1 / null=2')
    fig.suptitle(f'M2 pilot step {step}: held-out training episode')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
    np.savez_compressed(path.with_suffix('.npz'),predicted_pose=pred,true_pose=gt,
                        predicted_address=addresses,true_address=labels,address_valid=valid)
