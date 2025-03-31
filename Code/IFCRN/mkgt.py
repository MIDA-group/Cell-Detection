"""
 Generate "Ground Truth" masks from CytoBrowser Annotation.json file
 
 Place a Gauss-blob at each "marker" inside any "rectangle" (overlapping rectangles will lead to duplicated tiles)
"""

import os
from pathlib import Path
import argparse
import json
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import shutil
import itertools

def get_args():
    parser = argparse.ArgumentParser(description='Generate "Ground Truth" masks from CytoBrowser Annotation.json file')
    parser.add_argument('input', metavar='ANNOTATIONS', nargs='+', help='Filenames of input annotations')
    parser.add_argument('--tile', '-t', metavar='T', type=int, help='tile-size', default=256, required=False)
    parser.add_argument('--bin', '-b', metavar='B', type=int, help='bin-size', default=4, required=False)
    parser.add_argument('--sigma', '-s', metavar='SIGMA', type=int, help='Gaussian blur sigma', default=3, required=False)
    parser.add_argument('--output', '-o', metavar='OUTDIR', help='Directory where to write tiles and masks', required=True)  
    parser.add_argument('--prefix', '-p', metavar='PREFIX', help='Filename prefix', default="", required=False)  
    parser.add_argument('--dzi', '-d', metavar='DZIDIR', help='DZI-directory at the right scale', required=False)
    parser.add_argument('--class', '-c', dest='mclass', metavar='CLASS', nargs='+', help='Classes to include as positive', required=False) #Syntax: '-c class1 class2' => [class1 class2]
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()  

    total_tilecount = 0
    total_markercount = 0
    for i in args.input:
        print(f'\nProcessing file "{i}"')

        f = open(i)
        data = json.load(f)
        f.close()

        #rectangles in the annotation file
        rects = [d['points'] for d in data['annotations'] if len(d['points']) == 4]
        
        #x,y of markers
        markers = [list(d['points'][0].values()) for d in data['annotations'] if len(d['points']) == 1 and ((args.mclass is None) or (d['mclass'] in args.mclass))]
        points = np.array(markers)
        points /= args.bin

        #tile and within-tile position
        p_tile,p_pos=np.divmod(points,args.tile)

        #create output directories
        Path(args.output).mkdir(exist_ok=True)
        Path(os.path.join(args.output,'mask')).mkdir(exist_ok=True)
        if args.dzi: Path(os.path.join(args.output,'tile')).mkdir(exist_ok=True)

        #reciprocal scale factor, to reach peak-value = 1.0
        peakval = gaussian_filter(np.ones([1,1]), sigma=args.sigma, mode='constant').max()

        #for all rectangles in the annotation file
        for i,r in enumerate(rects):
            vertices = np.array([list(p.values()) for p in r])
            ul = vertices.min(axis=0)
            dr = vertices.max(axis=0)

            with np.printoptions(precision=1):
                print(f'Processing rectangle #{i}: [{ul}, {dr}] ',end='')

            #assert that quadrilateral really is rectangle
            for corner,dim in itertools.product([ul,dr],[0,1]):
                assert np.count_nonzero(corner[dim]==vertices[:,dim]) == 2, "Annotation quadrilateral is not axis aligned rectangle"

            #bin-factor
            ul /= args.bin
            dr /= args.bin

            #tiles completely inside rect
            ul=np.ceil(ul/args.tile).astype('int')
            dr=np.floor(dr/args.tile).astype('int')

            print(f'containing {np.prod(dr-ul)} tiles.')

            #for each tile in rect
            tilecount = 0
            markercount = 0
            for y in range(ul[1],dr[1]):
                for x in range(ul[0],dr[0]):
                    #print(f'{x}_{y}')

                    mask = np.zeros([args.tile,args.tile])
                    changed = False

                    # markers in current tile
                    pp=[pp.astype('int') for pt,pp in zip(p_tile,p_pos) if np.array_equal(pt,np.array([x,y]))]
                    for p in pp: 
                        mask[p[1],p[0]] = 1/peakval #y,x
                        changed = True
                        markercount += 1
                        
                    if changed:
                        mask = gaussian_filter(mask, sigma=args.sigma, mode='constant')
                        mask = np.clip(mask,0,1)
                   
                    Image.fromarray((mask * 255).astype(np.uint8)).save(f'{args.output}/mask/{args.prefix}{x}_{y}.jpg')
                    tilecount += 1

                    if args.dzi:
                        shutil.copy2(f'{args.dzi}/{x}_{y}.jpg',f'{args.output}/tile/{args.prefix}{x}_{y}.jpg')
            print(f'Wrote {markercount} markers in {tilecount} mask tiles.')
            total_tilecount += tilecount
            total_markercount += markercount

    print(f'Done. {total_tilecount} GT-mask tiles generated with {total_markercount} markers.')


