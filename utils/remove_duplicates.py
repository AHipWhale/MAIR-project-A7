def remove_duplicates(input_file, output_file):
    '''
    removes the duplicate lines from the dataset
    '''

    read_lines = [] # to store lines that have been read
    unique_lines = [] # to store unique lines
    total_lines_in_source = 0

    # append the unique lines in a list
    with open(input_file, 'r') as fin:
        for line in fin:
            total_lines_in_source += 1
            if line not in unique_lines:
                unique_lines.append(line)

    # save the deduplicated lines to the specified output file
    with open(output_file, 'w') as fout:
        for line in unique_lines:
            fout.write(line)

    print(f'there are a total of {total_lines_in_source} lines')
    print(f'there are {len(unique_lines)} unique lines')

remove_duplicates(input_file='datasets/dialog_acts_lower.dat', output_file='datasets/dialog_acts_deduplicated.dat')


