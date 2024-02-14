import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parses a directory of HTML pages and check for links to other pages.
    Returns a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Returns a probability distribution over which page to visit next,
    given a current page.

    """
    # Will store the calculated probabilities of each page
    prob = {}
    links = []

    #get the list of links the page connects to
    links = corpus.get(page, [])

    for key in corpus.keys():
        prob[key] = 0

    count = len(links)

    for link in links:
        prob[link] = damping_factor/count

    for key in prob:
        prob[key] += (1 - damping_factor)/len(corpus.keys())

    return prob
    
def sample_pagerank(corpus, damping_factor, n):
    """
    Returns PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Returns a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1).
    """
    randompage = random.choice(list(corpus.keys()))
    
    pagerank = {key: 0 for key in corpus.keys()}  # Initialize pagerank dictionary

    pagerank[randompage] += 1 / n  # Increment the initial random page

    for _ in range(n - 1):
        randompage = sampling(corpus, randompage, damping_factor)
        pagerank[randompage] += 1 / n

    return pagerank

def sampling(corpus, randompage, damping_factor):
    """
    Perform a single sampling step.
    """
    transition_dict = transition_model(corpus, randompage, damping_factor)

    keys = list(transition_dict.keys())
    probabilities = list(transition_dict.values())

    # Choose a key based on the specified probabilities
    random_key = random.choices(keys, weights=probabilities, k=1)[0]

    return random_key


def iterate_pagerank(corpus, damping_factor):
    """
    Returns PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    N = len(corpus)
    pagerank = {key: 1 / N for key in corpus.keys()}
    convergence = False 

    threshold = 0.001

    while not convergence:
        new_pagerank = {}
        max_diff = 0

        for page in corpus:
            new_pagerank[page] = (1 - damping_factor) / N

            for linking_page, links in corpus.items():
                if page in links:
                    new_pagerank[page] += damping_factor * pagerank[linking_page] / len(links)
        
        max_diff = max(max_diff, abs(new_pagerank[page] - pagerank[page]))

        if max_diff < threshold:
            convergence = True

        pagerank = new_pagerank
    
    pagerank = {page: rank / sum(pagerank.values()) for page, rank in pagerank.items()}

    return pagerank


if __name__ == "__main__":
    main()
