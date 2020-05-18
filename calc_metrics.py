from utils.metrics import calc_is_fid
from argparse import ArgumentParser



def main(args):

    is_mean, is_std, fid = calc_is_fid(args.original_path, args.generated_path,
                                       args.batch_size, args.n_split)
    
    print("Mean Inseption Score is: {} \n FID is: {}".format(str(is_mean), str(fid)))



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--original_path", type=str)
    parser.add_argument("--generated_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_split", type=int)

    args = parser.parse_args()

    main(args)

