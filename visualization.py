import matplotlib.pyplot as plt
import pandas as pd
import datetime
import argparse


def make_plot(data_path, stop_start, stop_end, data_start=None):
    data = pd.read_csv(data_path)
    if data_start is not None:
        data = data.iloc[list(data['date']).index(data_start):, :]

    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    fig = plt.figure(figsize=(10, 4))
    plt.axvspan(stop_start, stop_end, alpha=0.2, color='red')
    plt.axvspan(stop_start - datetime.timedelta(days=6 / 24), stop_start, alpha=0.2, color='green')
    plt.scatter(data.index, data['anomaly_score'], c='black', s=0.5)

    plt.xlabel('Date')
    plt.ylabel('Anomaly score')
    plt.margins(x=0)

    if data_start is not None:
        save_path = data_path.split('.csv')[0] + '_tight.png'
    else:
        save_path = data_path.split('.csv')[0] + '.png'
    plt.savefig(fname=save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        help='csv file path with columns [date, anomaly_score]')
    parser.add_argument('--fault-idx', type=int,
                        help='idx of unplanned fault')

    args, _ = parser.parse_known_args()

    config_dict = {0: [datetime.datetime(2018, 1, 25, 17, 47),
                       datetime.datetime(2018, 1, 25, 23, 7),
                       '2018-01-23 00:00:00'],
                   }

    stop_start, stop_end, data_start = config_dict[args.fault_idx]

    make_plot(args.data_path, stop_start, stop_end)
    make_plot(args.data_path, stop_start, stop_end, data_start)
