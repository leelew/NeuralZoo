import argparse


def get_lr_args():
    parse = argparse.ArgumentParser()

    # common default args
    parse.add_argument('--fit_intercept', type=bool, default=True)
    parse.add_argument('--normalize', type=bool, default=True)
    parse.add_argument('--max_iter', type=int, default=1000)
    parse.add_argument('--tol', type=float, default=1e-4)

    # ridge default args
    parse.add_argument('--alpha_ridge', type=float, default=1.0)
    parse.add_argument('--solver', type=str, default='auto')

    # ridge default args
    parse.add_argument('--alpha_lasso', type=float, default=0.01)

    # elasticnet default args
    parse.add_argument('--alpha_elasticnet', type=float, default=0.01)
    parse.add_argument('--l1_ratio', type=float, default=0.2)

    # common cv args
    parse.add_argument('--cv_num_folds', type=int, default=5)
    parse.add_argument('--cv_alphas', type=list, default=[0.1, 1.0, 10.0])

    # ridge cv args
    parse.add_argument('--cv_ridge', type=bool, default=True)

    # lasso cv args
    parse.add_argument('--cv_lasso', type=bool, default=False)

    # elasticnet cv args
    parse.add_argument('--cv_elasticnet', type=bool, default=False)

    return parse.parse_args()
