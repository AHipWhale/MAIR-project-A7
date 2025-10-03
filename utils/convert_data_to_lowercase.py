def convert_data_to_lowercase(input_file, output_file):
    '''
    read a file containing text. 
    convert all the text to lowercase.
    save the result in the output file
    '''

    with open(input_file, 'r') as f:
        text = f.read()

    text_lower = text.lower()

    with open(output_file, 'w') as f:
        f.write(text_lower)

    print(f'result saved to {output_file}')

convert_data_to_lowercase(input_file='datasets/dialog_acts.dat', output_file='datasets/dialog_acts_lower.dat')