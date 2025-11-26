import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
import os
import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm

np.random.seed(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_expanded_mask(segmented_image, expand_ratio=1.5):
    if segmented_image.dtype != np.uint8:
        segmented_image = segmented_image.astype(np.uint8)
    expanded_mask = np.zeros_like(segmented_image)
    contours, _ = cv2.findContours(
        segmented_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        new_w = int(w * expand_ratio)
        new_h = int(h * expand_ratio)
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        cv2.rectangle(expanded_mask,
                      (new_x, new_y),
                      (new_x + new_w, new_y + new_h),
                      255, -1)
    return expanded_mask

def postprocess_crf(args):
    files = os.listdir(args.sal_path)
    if (not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    expanded_output = None
    if getattr(args, 'generate_expanded', False):
        expanded_output = args.expanded_output_path
        if not os.path.exists(expanded_output):
            os.makedirs(expanded_output)
    for file in tqdm(files):
        img = cv2.imread(args.input_path+'/'+file, 1)
        annos = cv2.imread(args.sal_path+'/'+file, 0)
        annos = cv2.resize(annos, (img.shape[1], img.shape[0]))
        output = args.output_path+'/'+file
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], args.m)
        anno_norm = annos / 255.
        n_energy = -np.log((1.0 - anno_norm + args.epsilon)) / (args.tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + args.epsilon) / (args.tau * sigmoid(anno_norm))
        U = np.zeros((args.m, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=args.gaussian_sxy, compat=3)
        d.addPairwiseBilateral(sxy=args.bilateral_sxy, srgb=args.bilateral_srgb, rgbim=img, compat=5)
        Q = d.inference(1)
        map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        segmented_image = map.astype('uint8') * 255
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
        sizes = stats[:, cv2.CC_STAT_AREA]
        sorted_sizes = sorted(sizes[1:], reverse=True)
        top_k_sizes = sorted_sizes[:args.num_contours]
        im_result = np.zeros_like(im_with_separated_blobs)
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        segmented_image = im_result
        cv2.imwrite(output, segmented_image)
        if expanded_output:
            expanded_mask = generate_expanded_mask(segmented_image, args.expand_ratio)
            cv2.imwrite(os.path.join(expanded_output, file), expanded_mask)

def postprocess_thresholding(args):
    files = os.listdir(args.sal_path)
    if (not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    expanded_output = None
    if getattr(args, 'generate_expanded', False):
        expanded_output = args.expanded_output_path
        if not os.path.exists(expanded_output):
            os.makedirs(expanded_output)
    for file in tqdm(files):
        annos = cv2.imread(args.sal_path+'/'+file, 0)
        output = args.output_path+'/'+file
        annos = annos / 255.
        annos = (annos > args.threshold).astype(np.uint8)
        segmented_image = (annos > args.threshold).astype(np.uint8) * 255
        if getattr(args, 'filter', False) and args.num_contours > 0:
            nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(segmented_image)
            sizes = stats[:, cv2.CC_STAT_AREA]
            sorted_sizes = sorted(sizes[1:], reverse=True)
            top_k_sizes = sorted_sizes[:args.num_contours]
            im_result = np.zeros_like(im_with_separated_blobs)
            for index_blob in range(1, nb_blobs):
                if sizes[index_blob] in top_k_sizes:
                    im_result[im_with_separated_blobs == index_blob] = 255
            segmented_image = im_result
        cv2.imwrite(output, segmented_image)
        if expanded_output:
            expanded_mask = generate_expanded_mask(segmented_image, args.expand_ratio)
            cv2.imwrite(os.path.join(expanded_output, file), expanded_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Postprocess saliency maps')
    parser.add_argument('--postprocess', type=str, choices=['crf', 'thresholding'], required=True)
    parser.add_argument('--input-path', type=str, required=False, default='')
    parser.add_argument('--sal-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    # Thresholding options
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--num-contours', type=int, default=1)
    # CRF options
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--gaussian-sxy', type=float, default=3)
    parser.add_argument('--bilateral-sxy', type=float, default=50)
    parser.add_argument('--bilateral-srgb', type=float, default=5)
    # Expanded mask options
    parser.add_argument('--generate-expanded', action='store_true')
    parser.add_argument('--expanded-output-path', type=str, default='')
    parser.add_argument('--expand-ratio', type=float, default=1.0)

    args = parser.parse_args()
    if args.postprocess == 'crf':
        if not args.input_path:
            raise SystemExit('--input-path is required when using CRF')
        postprocess_crf(args)
    else:
        postprocess_thresholding(args)
