import socket
''' 
Part 1: how to send and receive data
# socket is an 'endpoint' that receives and sends information
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5) # queue of 5

while True:
	client_socket, address = s.accept() # Save the client socket object and their IP address
	# client_socket is essentially just another socket object like the object from socket.socket()
	print(f'Connection from {address} has been established!')
	client_socket.send(bytes('Welcome to the server!', 'utf-8')) # Send information to the client socket
	client_socket.close() # Close connection once the full message has been received
'''
'''
Part 2: How to send a message to the client when the message is greater than the buffer
HEADER_SIZE = 10

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((socket.gethostname(), 1234))
server_socket.listen(5)

while True:
	client_socket, address = server_socket.accept()
	print(f'Welcome to the server!')

	msg = 'Welcome to the server'
	msg = f'{len(msg):<{HEADER_SIZE}}'+msg

	client_socket.send(bytes(msg, 'utf-8'))
'''

