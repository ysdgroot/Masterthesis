import pickle


def save_model(model, name_dump_file):
    """
    Saving the python object/model
    :param model: an python object
    :param name_dump_file: String, with the name as filename
    :return: Boolean if the pickle dump succeeded
    """
    try:
        total_file_name = str(name_dump_file) + ".p"
        pickle.dump(model, open(total_file_name, "wb"))
        return True
    except IOError:
        return False


def get_model(name_file):
    """

    :param name_file: name of the pickle file to load
    :return: the object if there were no problems, otherwise None.
    """
    try:
        if name_file.endswith(".p"):
            file = name_file
        else:
            file = str(name_file) + ".p"
        obj = pickle.load(open(file, "rb"))
        return obj
    except IOError:
        print("Can not find file")
        return None
