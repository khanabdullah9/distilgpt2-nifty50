def obsolete(func):
    def wrapper(*args, **kwargs):
        print("The function is out of use.")
    return wrapper