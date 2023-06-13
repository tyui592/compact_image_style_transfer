# -*- coding: utf-8 -*-
"""Main app code.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import wandb
from arguments import get_args
from train import train_network
from evaluate import evaluate_network


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'train':
        run = wandb.init(
            project=args.wb_project,
            notes=args.wb_notes,
            name=args.wb_name,
            tags=args.wb_tags,
            config=args,
        )
        wandb.alert(title=args.wb_project,
                    text=f"{run.name} of {args.wb_project}: training start!")

        train_network(args)

        wandb.alert(title=args.wb_project,
                    text=f"{run.name} of {args.wb_project}: training done!")
        wandb.finish()

    elif args.mode == 'eval':
        evaluate_network(args)

    elif args.mode == 'prune_eval':
        pass
