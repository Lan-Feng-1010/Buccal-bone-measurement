import argparse,sys


def main(argv=None):
    parser=argparse.ArgumentParser(description='Postoperative CBCT measurement from existing 3-D NIfTI segmentations')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Generate and measure a synthetic NIfTI example')
    demo.add_argument('--output',required=True);demo.add_argument('--plot',action='store_true')
    for name in ['inspect','prepare','measure']:
        p=sub.add_parser(name);p.add_argument('config');p.add_argument('--output',required=True)
        if name=='measure':p.add_argument('--plot',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':
            from .demo import run_demo
            r=run_demo(args.output,args.plot);print('Synthetic demo passed. Distances:',[x['distance_mm'] for x in r['measurements']])
        elif args.command=='inspect':
            from .pipeline import inspect_case
            r=inspect_case(args.config,args.output);print('Implant components:',len(r['implant_components']))
        elif args.command=='prepare':
            from .pipeline import prepare_case
            print('Prepared request:',prepare_case(args.config,args.output))
        elif args.command=='measure':
            from .pipeline import measure_case
            print('Result status:',measure_case(args.config,args.output,args.plot)['status'])
    except (ValueError,TypeError,KeyError,OSError) as exc:
        print(f'Cannot complete {args.command}: {exc}',file=sys.stderr)
        return 2
    return 0
