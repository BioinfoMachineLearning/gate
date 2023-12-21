import pandas as pd
import numpy as np
import json
import os

def find_best_training_curve_v5(runs, savepath, gap_thresold):
    loss_list = {}

    for i in range(len(runs)-1,-1,-1):
        run = runs[i]

        if run.state != "finished":
            continue

        run_json_file = os.path.join(savepath, 'jsons', run.name + '.json')
        # print(run_json_file)
        if not os.path.exists(run_json_file):
            continue

        config_list = {k: v for k,v in run.config.items() if not k.startswith('_')}

        if config_list['loss_fun'] not in loss_list:
            loss_list[config_list['loss_fun']] = []

        with open(run_json_file) as f:
            data = json.load(f)

        train_loss = np.array(data['train_loss'])
        # print(train_loss)
        valid_loss = np.array(data['valid_loss'])
        if len(train_loss) < 20:
            continue

        if len(train_loss) != len(valid_loss):
            print(f"{run.name}, train: {str(len(train_loss))}, val: {str(len(valid_loss))}")
            continue

        valid_mean_loss = np.array(data['valid_mean_loss'])
        valid_median_loss = np.array(data['valid_median_loss'])

        valid_mean_rank_loss = np.array(data['valid_mean_rank_loss'])
        valid_median_rank_loss = np.array(data['valid_median_rank_loss'])

        if abs(train_loss[np.argmin(valid_loss)] - np.min(valid_loss)) < gap_thresold[config_list['loss_fun']]:
            loss_list[config_list['loss_fun']] += [run.name + ';' + str(np.min(valid_loss)) + ';' +
                                                   config_list['check_pt_dir'].split('//')[-1] + ';' +
                                                   str(valid_mean_loss[np.argmin(valid_loss)]) + ';' +
                                                   str(valid_median_loss[np.argmin(valid_loss)]) + ';' +
                                                   str(valid_mean_rank_loss[np.argmin(valid_loss)]) + ';' +
                                                   str(valid_median_rank_loss[np.argmin(valid_loss)])]

    for loss_fun in loss_list:
        data_dict = {'name': [], 'ckptdir': [], 'valid_loss': [],
                    'val_target_mean_mse': [], 'val_target_median_mse': [],
                    'val_target_mean_ranking_loss': [], 'val_target_median_ranking_loss': []}
        print(loss_fun)
        for name in loss_list[loss_fun]:
            jobname, valid_loss, ckpt_path, valid_mean_loss, valid_median_loss, valid_mean_rank_loss, valid_median_rank_loss = name.split(';')
            data_dict['name'] += [jobname]
            data_dict['valid_loss'] += [valid_loss]
            data_dict['ckptdir'] += [ckpt_path]
            data_dict['val_target_mean_mse'] += [valid_mean_loss]
            data_dict['val_target_median_mse'] += [valid_median_loss]
            data_dict['val_target_mean_ranking_loss'] += [valid_mean_rank_loss]
            data_dict['val_target_median_ranking_loss'] += [valid_median_rank_loss]
            df = pd.DataFrame(data_dict)
            df = df.sort_values(by=['valid_loss'])
            df.reset_index(inplace=True)
            # print(df)
            df.to_csv(savepath + '/' + loss_fun + '.csv')


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    args = parser.parse_args()

    data_dict = {'name': [], 'ckptdir': [], 'valid_loss': [],
                    'val_target_mean_mse': [], 'val_target_median_mse': [],
                    'val_target_mean_ranking_loss': [], 'val_target_median_ranking_loss': []}

    for run in os.listdir(args.indir):
        run_json_file = os.path.join(args.indir, run, 'learning_curve.json')
        if not os.path.exists(run_json_file):
            continue

        data = None
        with open(run_json_file) as f:
            data = json.load(f)
        
        train_loss = np.array(data['train_loss'])
        valid_loss = np.array(data['valid_loss'])
        print(len(valid_loss))
        val_target_mean_mse = np.array(data['val_target_mean_mse'])
        print(len(val_target_mean_mse))
        val_target_median_mse = np.array(data['val_target_median_mse'])
        val_target_mean_ranking_loss = np.array(data['val_target_mean_ranking_loss'])
        val_target_median_ranking_loss = np.array(data['val_target_median_ranking_loss'])
        break

        data_dict['ckptdir'] += [ckpt_path]
        data_dict['valid_loss'] += [valid_loss]
        data_dict['val_target_mean_mse'] += [valid_mean_loss]
        data_dict['val_target_median_mse'] += [valid_median_loss]
        data_dict['val_target_mean_ranking_loss'] += [valid_mean_rank_loss]
        data_dict['val_target_median_ranking_loss'] += [valid_median_rank_loss]
        df = pd.DataFrame(data_dict)
        df = df.sort_values(by=['valid_loss'])
        df.reset_index(inplace=True)

        
    

if __name__ == '__main__':
    cli_main()