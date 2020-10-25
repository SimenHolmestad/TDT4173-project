"""Create plots for dataset. The input file needs to contain
preprocessed data.

Run script with:

python3 create_plots.py <file-path-to-data-file>

"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_review_length_distribution(review_lengths):
    longest_review = max(review_lengths)
    number_of_reviews_for_lengths = np.zeros(longest_review + 1)
    x_values = np.arange(longest_review + 1)

    for review_length in review_lengths:
        number_of_reviews_for_lengths[review_length] += 1

    plt.figure(figsize=(6.8, 4.8))
    plt.plot(x_values, number_of_reviews_for_lengths)
    plt.title("Number of reviews with certain number of words")
    plt.xlabel("Amount of words for review")
    plt.ylabel("Number of reviews")

    plt.savefig("Plots/Lengts_of_reviews.png", dpi=300)

    plt.clf()


def plot_rating_distribution(review_ratings):
    unique, counts = np.unique(review_ratings, return_counts=True)

    plt.bar(unique, counts)
    plt.title("Distribution of stars for reviews")
    plt.xlabel("Amount of stars for review")
    plt.ylabel("Number of reviews")
    plt.ylim(0, 4000000)

    plt.savefig("plots/Distribution_of_stars.png", dpi=300)

    plt.clf()


def print_distribution_of_review_lenghts(review_lengths):
    for x in range(0, 300, 10):
        number_of_reviews_below_x = 0
        for review_length in review_lengths:
            if (review_length < x):
                number_of_reviews_below_x += 1
        percentage = round((number_of_reviews_below_x / len(review_lengths)) * 100, 2)
        print("Percentage of reviews with less than", x, "words is", percentage)


def main():
    if len(sys.argv) < 2:
        print("You have to specify a filename for the input file")
        print("You should run script as:")
        print("python3 create_plots.py <file-path-to-data-file>")
        sys.exit()

    filepath = sys.argv[1]

    max_number_of_lines = 10000000000000

    review_lengths = []
    review_ratings = []

    # Extract content from file
    print("reading content of {}...".format(filepath))
    cnt = 0
    with open(filepath) as fp:
        line = fp.readline()  # Top of csv-file
        line = fp.readline()
        while line and cnt < max_number_of_lines:
            cnt += 1

            training, label = line.split("\", ")
            label = float(label)

            review_ratings.append(label)
            number_of_words = len(training.split(" "))
            review_lengths.append(number_of_words)

            # Read next line
            line = fp.readline()

    # Make plots
    print("Plotting review length distribution")
    plot_review_length_distribution(review_lengths)

    print("Plotting review rating distribution")
    plot_rating_distribution(review_ratings)

    # Print review length distribution
    print("\nDistribution of review lengths:")
    print_distribution_of_review_lenghts(review_lengths)


if __name__ == "__main__":
    main()
