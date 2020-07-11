from utils import dict_to_tensor_IO

def main():
    dict_data = {'a': 10, 'b': 20, 'c': 30, 'd': 40}
    converter = dict_to_tensor_IO(dict_data=dict_data)
    test_tensor = converter.map_dict_state_point_to_tensor(dict_data=dict_data)
    print(test_tensor)

    test_dict = converter.map_tensor_state_point_to_dict(test_tensor)
    print(test_dict)

if __name__ == "__main__":
    main()