import os
import random
import re
import sys
# from queue import Queue
# from collections import defaultdict

DAMPING = 0.852
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
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
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
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    likehood = {}
    pagenums = len(corpus)
    linknums = len(corpus[page])
    if linknums <= 0:
        for key in corpus.keys():
            likehood[key] = 1 / pagenums
        return likehood
    else:
        for key in corpus.keys():
            likehood[key] = ((1 - damping_factor) * 1) / pagenums
        for linkpages in corpus[page]:
            likehood[linkpages] += damping_factor / linknums
        return likehood

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pagerank = {page : 0 for page in corpus}
    cur_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        pagerank[cur_page] += 1
        transition = transition_model(corpus, cur_page, damping_factor)
        cur_page = random.choices(list(corpus.keys()), weights = transition.values(), k = 1)[0]

    total_times = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total_times

    return pagerank

    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    """
    # wrong idea! using topological sort can not solve this kind of problems
    # topological sort is appropriate for DAG, namely directed acyclic graph
    # this problem can not be seen as a DAG

    pagenums = len(corpus)
    pagerank = {}
    InDegree = {}
    linkpages = defaultdict(list)
    ZeroInDegree = Queue()
    for key, value in corpus.items():
        InDegree[key] = len(corpus[key])
        pagerank[key] = (1 * (1.0 - damping_factor)) / pagenums
        if InDegree[key] == 0:
            ZeroInDegree.put(key)
            continue
        for page in value:
            linkpages[page].append(key)
    
    while not ZeroInDegree.empty():
        page = ZeroInDegree.get()
        nums = len(corpus[page])
        for link in corpus[page]:
            pagerank[page] += (damping_factor * pagerank[link]) / nums
        for link in linkpages[page]:
            InDegree[link] -= 1
            if InDegree[link] <= 0:
                ZeroInDegree.put(link)

    return pagerank
    """

    num_pages = len(corpus)
    # if it not linked with any page, all pages will add 1 / page_nums
    pagerank = {page: 1 / num_pages for page in corpus}    # create a dict
    threshold = 0.0001                                     # Convergence threshold
    change = float('inf')

    while change > threshold:
        new_pagerank = {}
        total_diff = 0

        for page in corpus:
            new_pagerank[page] = (1 - damping_factor) / num_pages
        
            for linking_page in corpus:
                if page in corpus[linking_page]:
                    new_pagerank[page] += damping_factor * (pagerank[linking_page]) / len(corpus[linking_page])

            total_diff += abs(new_pagerank[page] - pagerank[page])

        pagerank = new_pagerank
        change = total_diff
        # print(pagerank)
    
    total = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total
    
    return pagerank

    raise NotImplementedError


if __name__ == "__main__":
    main()
