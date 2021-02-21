
import argparse


def get_parse():
    parse = argparse.ArgumentParser()

    # data_loader parse
    parse.add_argument('raw_path', type=str, default='/hard/lilu/ERA5/')
    parse.add_argument('preliminary_path', type=str,
                       default='/hard/lilu/ERA5/preliminary/')
    parse.add_argument('save_name', type=str, default='t2m')
    parse.add_argument('NLAT', type=int, default=180)
    parse.add_argument('NLON', type=int, default=360)

    # data generator parse
    parse.add_argument('input_path', type=str,
                       default='/hard/lilu/ERA5/inputs/')
    parse.add_argument('intervel', type=int, default=18)
    parse.add_argument('len_inputs', type=int, default=10)
    parse.add_argument('window_size', type=int, default=7)

    # train
    parse.add_argument('--mdl_name', type=str, default='ml.lr.ridge')
    parse.add_argument('--model_path', type=str,
                       default='/hard/lilu/ERA5/save/')
    parse.add_argument('--forecast_path', type=str,
                       default='/hard/lilu/ERA5/forecast/')
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--epochs', type=int, default=50)
    parse.add_argument('--num_jobs', type=int, default=0)

    # inference
    parse.add_argument('--score_path', type=str,
                       default='/hard/lilu/ERA5/score/')

    return parse.parse_args()
