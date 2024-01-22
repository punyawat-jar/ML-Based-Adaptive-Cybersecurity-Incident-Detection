import tqdm

def progress_bar(*args, **kwargs):
    bar = tqdm(*args, **kwargs)

    def checker(x):
        bar.update(1)
        return False

    return checker