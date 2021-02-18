import socket
import select

HEADER_LENGTH = 10
IP = '127.0.0.1'
PORT = 1234

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allows for reconnection
server_socket.bind((IP,PORT))
server_socket.listen()
sockets_list = [server_socket]
clients = {} # Save client information (address and username)

def receive_message(client_socket):
	try:
		message_header = client_socket.recv(HEADER_LENGTH)

		if not len(message_header): # no data received (ie client closed connection)
			return False

		message_length = int(message_header.decode('utf-8'))
		return{"header": message_header, "data": client_socket.recv(message_length)}


	except:
		return False

while True:
	read_sockets, __, exception_sockets = select.select(sockets_list, [], sockets_list)

	for notified_socket in read_sockets:
		if notified_socket == server_socket: # ie: someone just connected
			client_socket, client_address = server_socket.accept()

			user = receive_message(client_socket)
			if user is False: # Someone just disconnected
				continue

			sockets_list.append(client_socket) # Add new client socket to list
			clients[client_socket] = user # save the username as the value to the key that is the socket object
			print(f"Accepted new connection from {client_address[0]}:{client_address[1]} username: {user['data'].decode('utf-8')}")

		else:
			message = receive_message(notified_socket)

			if message is False: # Someone disconnected
				# print(f"Closed connection from {clients[notified_socket]['data'].decode('utf-8')}")
				print(clients)
				print('Closed connection from: {}'.format(clients[notified_socket]['data'].decode('utf-8')))
				# Clean up
				sockets_list.remove(notified_socket)
				del clients[notified_socket]
				continue

			user = clients[notified_socket]
			print(f"Received message from {user['data'].decode('utf-8')}: {message['data'].decode('utf-8')}")

			for client_socket in clients: # Send message to other users
				if client_socket != notified_socket: # Don't send message to sender
					client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])

			for notified_socket in exception_sockets:
				socket_list.remove(notified_socket)
				del clients[notified_socket]