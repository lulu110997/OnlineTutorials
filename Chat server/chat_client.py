# https://pythonprogramming.net/client-chatroom-sockets-tutorial-python-3/
import socket
import select
import errno
import sys

HEADER_LENGTH = 10
IP = '127.0.0.1'
PORT = 1234

client_username = input('Username: ')
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))
client_socket.setblocking(False) # Sets the receive funcitonality to be non-blocking

username = client_username.encode('utf-8') # Send username info to the chat server
username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
client_socket.send(username_header + username)

while True:
	# Send and receive messages from chat server
	message = input(f"{client_username} > ")

	if message: # In case someone just presses enter
		message = message.encode('utf-8')
		message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')
		client_socket.send(message_header + message) # Send to self

	
	# Receive things until we hit an error
	try:
		while True:
			username_header = client_socket.recv(HEADER_LENGTH)
			if not len(username_header):
				# We got no data
				print('Connection closed by the server')
				sys.exit()

			# Obtain message and username 
			username_length = int(username_header.decode('utf-8'))
			username = client_socket.recv(username_length).decode('utf-8')

			message_header = client_socket.recv(HEADER_LENGTH)
			message_length = int(message_header.decode('utf-8'))
			message = client_socket.recv(message_length).decode('utf-8')

			print(f"{username} > {message}")

	except IOError as e:
		# This is normal on non blocking connections - when there are no incoming data error is going to be raised
        # Some operating systems will indicate that using AGAIN, and some using WOULDBLOCK error code
        # We are going to check for both - if one of them - that's expected, means no incoming data, continue as normal
        # If we got different error code - something happened
			if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
				print('Reading error', str(e))
				sys.exit()
			continue # Cause nothing happened

	except Exception as e:
			print('General error',str(e))
			sys.exit()