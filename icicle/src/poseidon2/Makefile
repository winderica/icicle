test_poseidon: test.cu 
	nvcc -o test_poseidon -I../../include -DFIELD=bn254 -DFIELD_ID=1 -DCURVE_ID=1 extern.cu test.cu

test_poseidon_m31: test_m31.cu 
	nvcc -o test_poseidon_m31 -I../../include -DFIELD=m31 -DFIELD_ID=1003 extern.cu test_m31.cu