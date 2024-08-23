def bubble_sort(num_list):
    """
    Sorts a list of numbers using the bubble sort algorithm in ascending order.

    Args:
        num_list (list): A list of numbers to be sorted.

    Returns:
        list: The sorted list of numbers in ascending order.
    """
    n = len(num_list)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            if num_list[j] > num_list[j+1]:
                num_list[j], num_list[j+1] = num_list[j+1], num_list[j]    
                return num_list

# Example usage:
input_list = [64, 34, 25, 12, 22, 11, 90]
print("Original list:", input_list)
sorted_list = bubble_sort(input_list)
print("Sorted list:", sorted_list)