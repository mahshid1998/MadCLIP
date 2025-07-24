import os
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from sklearn.metrics import roc_auc_score
from loss import  Loss_detection
from Prompt.promptChooser import PromptChooser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='MadCLIP training')
    parser.add_argument('--exp_name', type=str, help='name it')

    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help='ViT-B-16-plus-240, ViT-L-14-336')
    parser.add_argument('--pretrain', type=str, default='openai', help='laion400m, openai')

    parser.add_argument('--obj', type=str, default='Brain', help='target dataset, it should be in CLASS_INDEX')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', action="store_true", default=True)
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=60, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24],
                        help="Layers to apply adapter on")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=16, help="number of samples for training")
    parser.add_argument('--iterate', type=int, default=0)

    parser.add_argument('--text_mood', type=str, default='learnable_all',
                        help=" the prompt strategy: fix, learnable_all, learnable_abnormal")
    parser.add_argument('--contrast_mood', type=str, default='yes', help="no, yes")
    # parser.add_argument('--adapter_res_mood', type=float, default=0.0, help="whether to apply residual after adapters, 0.0(for no residual), 0.2,0.4,0.8... or any value < 1")

    parser.add_argument('--dec_type', type=str, default='mean', help="aggregation of adapters outputs: mean, max, both,"
                                                                     " they account for global or localized decisions")
    parser.add_argument('--loss_type', type=str, default='sigmoid', help="Loss type: softmax, sigmoid")

    parser.add_argument('--visionA', type=str, default="MFCFC", help="Type of adapter layers")



    args = parser.parse_args()
    setup_seed(args.seed)
    from CLIP.adapter_shared import CLIP_Inplanted


    ############################
    # Instantiating CLIP model #
    ############################
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                             pretrained=args.pretrain, require_pretrained=True)
    clip_model.to(device)
    clip_model.eval()

    # add adapters on CLIP
    model = CLIP_Inplanted(args, clip_model=clip_model).to(device)
    model.eval()

    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    loss_det = Loss_detection(args= args, device=device, loss_type=args.loss_type,dec_type=args.dec_type, lr=args.learning_rate)



    # text maker
    text_chooser = PromptChooser(clip_model, args, device)


   # loading the saved checkpoint
    checkpoint = torch.load(os.path.join(args.save_path, f'{args.obj}.pth'))
    print(os.path.join(args.save_path, f'{args.obj}.pth'))
    model.normal_det_adapters.load_state_dict(checkpoint["normal_det_adapters"])
    model.abnormal_det_adapters.load_state_dict(checkpoint["abnormal_det_adapters"])

    # Load text chooser parameters based on text_mood
    if text_chooser.text_mood == 'fix':
        if 'text_features_fix' in checkpoint:
            text_chooser.text_features_fix = checkpoint['text_features_fix']
        else:
            raise KeyError("Missing 'text_features_fix' in checkpoint for text_mood='fix'")
    elif text_chooser.text_mood == 'learnable_all':
        if 'prompt_maker_normal' in checkpoint and 'prompt_maker_abnormal' in checkpoint:
            text_chooser.prompt_maker_normal.load_state_dict(checkpoint['prompt_maker_normal'])
            text_chooser.prompt_maker_abnormal.load_state_dict(checkpoint['prompt_maker_abnormal'])
        else:
            raise KeyError(
                "Missing 'prompt_maker_normal' or 'prompt_maker_abnormal' in checkpoint for text_mood='learnable_all'")

    else:  # only learnable abnormal
        if 'prompt_maker_abnormal' in checkpoint:
            text_chooser.prompt_maker_abnormal.load_state_dict(checkpoint['prompt_maker_abnormal'])
        else:
            raise KeyError("Missing 'prompt_maker_abnormal' in checkpoint for text_mood='only learnable abnormal'")



    result, auc, pauc, sim_text = test(args, model, test_loader, text_chooser, loss_det)


def test(args, model, test_loader, text_chooser, loss_det):
    image_list = []
    gt_list = []
    gt_mask_list = []
    det_final = []
    seg_final = []

    with torch.no_grad():
        text_features = text_chooser()  # [768,2]
        sim_text = F.cosine_similarity(text_features[:, 0], text_features[:, 1], dim=0)

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, det_model, seg_model = model(image, text_features)

            if CLASS_INDEX[args.obj] > 0:
                # seg head
                anomaly_maps = []
                for layer in range(len(seg_model)):
                    seg_scores_cur = seg_model[layer] # batch, 2,240,240
                    seg_scores_cur = loss_det.sync_AS(seg_scores_cur)
                    anomaly_map = 0.5 * (1- seg_scores_cur[:,0,:,:]) + 0.5 * seg_scores_cur[:,1,:,:] # batch, 240,240
                    anomaly_maps.append(anomaly_map.cpu().numpy())

                score_map = np.sum(np.stack(anomaly_maps), axis=0) # summing all anomaly_map of all layer [batch size, 240,240]
                seg_final.extend(score_map)
            #else:
            # det head
            anomaly_scores_all = 0
            for layer in range(len(det_model)):
                det_scores_cur = det_model[layer] # [batch, 289,2]
                anomaly_scores_all += loss_det.validation(det_scores_cur)

            det_final.extend(anomaly_scores_all.cpu().numpy())

            gt_mask_list.extend(mask.squeeze(1).cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            image_list.extend(image.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    # image_list = np.asarray(image_list)

    if CLASS_INDEX[args.obj] > 0:
        seg_scores = np.array(seg_final)
        seg_scores=  (seg_scores - seg_scores.min()) / ( 1e-4+
                    seg_scores.max() - seg_scores.min())
        # compute pixel level auc
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), seg_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')

        # Reshape segment scores for image-level evaluation
        segment_scores_flatten = seg_scores.reshape(seg_scores.shape[0], -1)
        # Compute image-level ROC AUC
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')
        return seg_roc_auc + roc_auc_im, roc_auc_im, seg_roc_auc, sim_text

    else:
        det_image_scores_zero = np.array(det_final)
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / ( 1e-4 + det_image_scores_zero.max() - det_image_scores_zero.min())
        img_roc_auc_det = roc_auc_score(gt_list, det_image_scores_zero)
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)} from det')
        return img_roc_auc_det, img_roc_auc_det , img_roc_auc_det, sim_text

if __name__ == '__main__':
    main()


