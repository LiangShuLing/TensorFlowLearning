import primes
import primes_python
import primes_python_cy
import HelloWorld
import time

time_start=time.time()
print(primes.primes(1000))
time_end=time.time()
print('totally cost',(time_end-time_start))

time_start=time.time()
print(primes_python.primes_python(1000))
time_end=time.time()
print('totally cost',time_end-time_start)

time_start=time.time()
print(primes_python_cy.primes_python(1000))
time_end=time.time()
print('totally cost',time_end-time_start)
