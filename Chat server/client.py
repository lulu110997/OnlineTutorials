import socket
''' 
Part 1: how to send and receive data
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))

# The maximum amount of data to be received at 
# once is specified by the recv method's arg
msg = s.recv(1024) 
print(msg.decode('utf=8'))

# The method below shows one way that allows the user to have a small buffer but receive
# a message that exceeds the bufer size. Using this method will require the client socket 
# connection to be closed by the server
full_msg = ''
while True:
	msg = s.recv(8)
	if len(msg) <= 0:
		break
	full_msg += msg.decode('utf-8')
print(full_msg)
'''
'''
Part 2: How to send a message to the client when the message is greater than the buffer
HEADER_SIZE = 10

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((socket.gethostname(), 1234))

while True:
	full_msg = ''
	new_msg = True # Track the number of new messages we receive
	
	while True:
		msg = client_socket.recv(16)
		if new_msg:
			print(f'new message length: {msg[:HEADER_SIZE]}')
			msg_len = int(msg[:HEADER_SIZE])
			new_msg = False
		full_msg += msg.decode('utf-8')

		if len(full_msg) - HEADER_SIZE == msg_len:
			print("Full message received!")
			print(full_msg[HEADER_SIZE:])
			new_msg = True
			full_msg = ''
'''