import os
import yaml


def dataset_parser(args):
    project_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    with open(f'{project_dir}/dataset/dataset_conf.yaml') as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)
        conf = conf[args.dataset]
    img_dir = conf['image_dir']
    img_dir = img_dir + f'/{args.align}_aligned'
    targets_txt = conf['targets_txt']

    with open(targets_txt, 'r') as fp:
        lines = fp.readlines()
    target_list = [l.strip() for l in lines]

    if args.dataset == 'cfp-fp-F' or args.dataset == 'cfp-fp-P' or \
            args.dataset == 'cfp-fp-200-F' or args.dataset == 'cfp-fp-200-P':
        mode = args.dataset[-1]  # F or P
        protocol_dir = conf['protocol_dir']
        with open(protocol_dir + f'/Pair_list_{mode}.txt', 'r') as fp:
            lines = fp.readlines()
        idx_dict = {}
        for line in lines:
            num, path = line.strip().split()
            plist = path.split('/')
            plist[2] = f'{args.align}_aligned'
            path = '/'.join(plist)
            idx_dict[num] = protocol_dir + '/' + path

    targets = []
    imgdirs = []
    for target in target_list:
        if args.dataset == 'lfw' or args.dataset == 'lfw-200':
            target_name = target.split('/')[-1][:-9]
            imgdir = os.path.join(img_dir, target_name, target)
        elif args.dataset == 'cfp-fp-F' or args.dataset == 'cfp-fp-P' or \
                  args.dataset == 'cfp-fp-200-F' or args.dataset == 'cfp-fp-200-P':
            imgdir = idx_dict[target]
            target = target + '.jpg'
        elif args.dataset == 'colorferet-dup1' or args.dataset == 'colorferet-dup2':
            tokens = target.split(' ')
            imgdir = os.path.join(img_dir, tokens[0], tokens[1])
            target = tokens[1]
        else:
            raise NotImplementedError(f'dataset {args.dataset} is not implemented!')
        targets.append(target)
        imgdirs.append(imgdir)

    return targets, imgdirs