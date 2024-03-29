# Copyright (C) 2022 Luis Hartmann and Fabio Panduri
# This file is part of matura.
# matura is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.
# matura is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with matura. If not, see <https://www.gnu.org/licenses/>.
"""
Used to plot data generated by the algorithms
"""
import argparse
import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def args() -> 'argparse':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v', '--verbose', help="Print information about the sample", action="store_true")
    parser.add_argument(
        '-f', '--plot-fitness', help="Plot the fitness ", action="store_true")
    parser.add_argument(
        '-fs', '--plot-fitness-std', help="Plot the standard deviation the fitness values", action="store_true")
    parser.add_argument(
        '-t', '--plot-time', help="Plot the time", action="store_true")
    parser.add_argument(
        '-ts', '--plot-time-std', help="Plot the standard deviation of the time", action="store_true")
    parser.add_argument(
        '-ti', '--plot-time-int', help="Plot the accumulated time", action="store_true")
    parser.add_argument(
        '-pr', '--plot-prediction', help="Plot the prediction (only for sgd)", action="store_true")
    parser.add_argument(
        '-ps', '--plot-prediction-std',
        help="Plot the standard deviation of the prediction (only for sgd)", action="store_true")
    parser.add_argument(
        '-p', '--path', help="Specify the folder", type=str, required=True)

    m_g = parser.add_mutually_exclusive_group()
    m_g.add_argument(
        '-d', '--date_range', help="Specify the date range in format 'YYYY-MM-DD-hh-mm-ss--YYYY-MM-DD-hh-mm-ss'",
        type=str)
    m_g.add_argument(
        '-s', '--single-file', help="Specify the file", type=str)

    return parser.parse_args()


def read_data_file(filename):
    with open(filename, "r") as f:
        data = json.loads(f.read())
    return data


def plot(label, x_label, y_label, *data):
    for i, data_set in enumerate(data):
        x = list(range(0, len(data_set)))
        y = data_set

        if label:
            plt.plot(x, y, label=label[i])
        else:
            plt.plot(x, y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if label:
        plt.legend()
    plt.show()


def prefix_sum(arr):
    prefix_sum_arr = [arr[0]]
    for element in arr[1:]:
        prefix_sum_arr.append(element + prefix_sum_arr[-1])

    return prefix_sum_arr


def main():
    relevant_data = [
        "performance history",
        "time history",
        "generation time history",
        "best fitness history",
        "fitness history",
        "loss",
        "prediction"
    ]

    arguments = args()

    if arguments.single_file:
        data = read_data_file(arguments.single_file)

    else:
        dates = arguments.date_range.split('--')
        date_0 = datetime.datetime.strptime(dates[0], "%Y-%m-%d-%H-%M-%S")
        date_1 = datetime.datetime.strptime(dates[1], "%Y-%m-%d-%H-%M-%S")

        blank_data = None

        count = 0
        for filename in sorted(list(os.listdir(arguments.path))):
            file_data = read_data_file(f"{arguments.path}/{filename}")
            date = datetime.datetime.strptime(
                file_data["time"], "%Y-%m-%d-%H-%M-%S")

            if date_0 <= date and date <= date_1:
                if not blank_data:
                    blank_data = file_data.copy()
                    for l in relevant_data:
                        if l in blank_data.keys():
                            blank_data[l] = [[]
                                             for _ in range(len(blank_data[l]))]

                count += 1

                # summarize the data for plotting
                if arguments.plot_fitness:
                    if "performance history" in file_data.keys():
                        for i, x in enumerate(file_data["performance history"]):
                            blank_data["performance history"][i].append(x)
                    elif "loss" in file_data.keys():
                        for i, x in enumerate(file_data["loss"]):
                            blank_data["loss"][i].append(x)
                    else:
                        for i, x in enumerate(file_data["best fitness history"]):
                            blank_data["best fitness history"][i].append(x)
                        for i, x in enumerate(file_data["fitness history"]):
                            blank_data["fitness history"][i].append(x)

                if arguments.plot_time:
                    if "time history" in file_data.keys():
                        for i, x in enumerate(file_data["time history"]):
                            blank_data["time history"][i].append(x)
                    else:
                        for i, x in enumerate(file_data["generation time history"]):
                            blank_data["generation time history"][i].append(x)

                if arguments.plot_prediction:
                    for i, x in enumerate(file_data["prediction"]):
                        blank_data["prediction"][i].append(x)

        if not blank_data:
            print("[ERROR] No files found")
            sys.exit()

        for l in relevant_data:
            if l in blank_data.keys():
                summed_data = []
                for d in blank_data[l]:
                    summed_data.append(sum(d))
                average_data = list(map(lambda x: x/count, summed_data))

                standard_deviation = []
                for t_step in blank_data[l]:
                    standard_deviation.append(np.std(t_step))

                blank_data[l] = average_data
                blank_data[f"{l} std"] = standard_deviation

        if arguments.verbose:
            print(f"[INFO] Sample size: {count}")

        if arguments.plot_fitness:
            if "performance history" in blank_data.keys():
                plot(None, "Episodes", "Fitness",
                     blank_data["performance history"])
            elif "loss" in blank_data.keys():
                plot(None, "Epochs", "Loss function", blank_data["loss"])
            else:
                plot(["best fitness history", "average fitness history"],
                     "Generations", "Fitness", blank_data["best fitness history"], blank_data["fitness history"])

        if arguments.plot_fitness_std:
            if "performance history" in blank_data.keys():
                plot(None, "Episodes", "Fitness std.",
                     blank_data["performance history std"])
            elif "loss" in blank_data.keys():
                plot(None, "Epochs", "Loss std.",
                     blank_data["loss std"])
            else:
                plot(["best fitness history std", "average fitness history std"],
                     "Generations", "Fitness std.",
                     blank_data["best fitness history std"], blank_data["fitness history std"])

        if arguments.plot_time:
            if "time history" in blank_data.keys():
                x_label = "Epochs" if "loss" in blank_data.keys() else "Episodes"
                plot(None, x_label, "Time in s", blank_data["time history"])
            else:
                plot(None, "Generations", "Time in s",
                     blank_data["generation time history"])

        if arguments.plot_time_std:
            if "time history" in blank_data.keys():
                x_label = "Epochs" if "loss" in blank_data.keys() else "Episodes"
                plot(None, x_label, "Time std. in s",
                     blank_data["time history std"])
            else:
                plot(None, "Generations", "Time std. in s",
                     blank_data["generation time history std"])

        if arguments.plot_prediction:
            plot(None, "x", "f(x)", blank_data["prediction"])
        if arguments.plot_prediction_std:
            plot(None, "x", "f(x) std", blank_data["prediction std"])

        if arguments.plot_time_int:
            if "performance history" in blank_data.keys():
                time_int = prefix_sum(blank_data["time history"])
                x_label = "Episodes"
            else:
                time_int = prefix_sum(blank_data["generation time history"])
                x_label = "Generations"

            plot(None, x_label, "Accumulated time in s", time_int)


if __name__ == "__main__":
    main()
