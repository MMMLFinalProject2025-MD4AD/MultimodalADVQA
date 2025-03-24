import argparse
import os
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval


def write_summary(metrics_summary, out_file):
    with open(out_file, 'w') as f:
        f.write(f"mAP: {metrics_summary['mean_ap']:.4f}\n")
        err_map = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE',
        }
        for k, label in err_map.items():
            f.write(f"{label}: {metrics_summary['tp_errors'][k]:.4f}\n")
        f.write(f"NDS: {metrics_summary['nd_score']:.4f}\n")
        f.write(f"Eval time: {metrics_summary['eval_time']:.1f}s\n\n")

        f.write("Per-class results:\n")
        f.write("%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\n" %
                ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
        for class_name, ap in metrics_summary['mean_dist_aps'].items():
            tp = metrics_summary['label_tp_errors'][class_name]
            f.write('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\n' % (
                class_name, ap,
                tp['trans_err'],
                tp['scale_err'],
                tp['orient_err'],
                tp['vel_err'],
                tp['attr_err'],
            ))


def main():
    parser = argparse.ArgumentParser(description='Evaluate and print nuScenes metrics from result json.')
    parser.add_argument('--data-root', required=True, help='Path to nuScenes dataset root.')
    parser.add_argument('--json', required=True, help='Path to results_nusc.json file.')
    parser.add_argument('--out', required=True, help='Path to save summary.txt file.')
    parser.add_argument('--eval-set', default='val', help='Evaluation split (default: val).')
    parser.add_argument('--version', default='v1.0-trainval', help='NuScenes version.')
    args = parser.parse_args()

    print("Running nuScenes official evaluation...")

    cfg = config_factory('detection_cvpr_2019')
    nusc = NuScenes(dataroot=args.data_root, version=args.version, verbose=False)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    evaluator = DetectionEval(
        nusc=nusc,
        config=cfg,
        result_path=args.json,
        eval_set=args.eval_set,
        output_dir=out_dir,
        verbose=True,
    )

    metrics_summary = evaluator.main(plot_examples=0, render_curves=False)

    # Print to file
    write_summary(metrics_summary, args.out)

    print(f"Evaluation summary written to: {args.out}")


if __name__ == '__main__':
    main()
