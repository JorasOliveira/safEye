import face_recognition
from time import time

start = time()

known_image = face_recognition.load_image_file("img/mat1.jpeg")
matheus_encoding = face_recognition.face_encodings(known_image)[0]

known_image_time = time()

mat = face_recognition.load_image_file("img/mat2.jpeg")
mat_encoding = face_recognition.face_encodings(mat)[0]
mat_results = face_recognition.compare_faces([matheus_encoding], mat_encoding)

end_mat = time()

joras = face_recognition.load_image_file("img/joras1.jpeg")
joras_encoding = face_recognition.face_encodings(joras)[0]
joras_results = face_recognition.compare_faces([matheus_encoding], joras_encoding)

end = time()

print("É o Matheus." if mat_results[0] else "Não é o Matheus.")
print("É o Matheus." if joras_results[0] else "Não é o Matheus.")
print(f"Tempo de execução: {end - start} segundos.")
print(f"Tempo de execução da primeira imagem: {known_image_time - start} segundos.")
print(f"Tempo de execução da segunda imagem: {end_mat - known_image_time} segundos.")
print(f"Tempo de execução da terceira imagem: {end - end_mat} segundos.")