#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bmp_util.c"

/* Riferimenti utili:
 * https://developer.nvidia.com/blog/even-easier-introduction-cuda/
 * https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 */

static void HandleError(cudaError_t err, const char *file, int line){ 
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE ); 
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define indexx(i, j, N)  ((i)*(N)) + (j)

__device__ void atomicMin(float* const address, const float value)
{
    if (*address <= value)
    {
        return;
    }
  
    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;
  
    do
    {
        assumed = old;
        if (__int_as_float(assumed) <= value)
        {
            break;
        }
  
        old = atomicCAS(addressAsI, assumed, __float_as_int(value));
    } while (assumed != old);
} 

__global__ void getMin(const float* __restrict__ input, const int size, const int sizeM, float* minOut, int* minIdxOut, int* minIdyOut)
{
    __shared__ float sharedMin;
    __shared__ int sharedMinIdx;
    __shared__ int sharedMinIdy;
  
    if (0 == threadIdx.x && threadIdx.y == 0)
    {
        sharedMin = input[threadIdx.x];
        sharedMinIdx = 0;
        sharedMinIdy = 0;
    }
  
    __syncthreads();
  
    float localMin = input[0];
    int localMinIdx = 0;
    int localMinIdy = 0;
  
    for (int i = blockIdx.y * blockDim.y + threadIdx.y ; i < size; i += blockDim.y)
    {  
        for (int j = threadIdx.x + blockIdx.x * blockDim.x; j < sizeM; j+=blockDim.x) {
            float val = input[i * sizeM + j];
        
            if (localMin > abs(val)){
                localMin = abs(val);
                localMinIdx = j;
                localMinIdy = i;
            }
        }    
    }
  
	// Funzione Atomica per il minimo con valori float
    
    atomicMin(&sharedMin, localMin);
  
    __syncthreads();
  
    if (sharedMin == localMin)
    {
        sharedMinIdx = localMinIdx;
        sharedMinIdy = localMinIdy;
    }
  
    __syncthreads();
  
    if (0 == threadIdx.x && threadIdx.y == 0)
    {
        minOut[blockIdx.x] = sharedMin;
        minIdxOut[blockIdx.x] = sharedMinIdx;
        minIdyOut[blockIdx.x] = sharedMinIdy;
    }
}

__global__ void getMatch(float *I, float *T, int Iw, int Ih, int Tw, int Th, float *differences) {

	float diff = 0;
	float temp;
	int  k, l;
	int i = 0;
	int j = 0;
	

	extern __shared__ float s_template[];

	// Copio l'immagine template in shared memory (impongo che la dimensione del template deve essere più piccola della shared memory, il test si effettua nel main)
	for (i = threadIdx.y; i <Th; i+=blockDim.y ) {
		for (j = threadIdx.x; j <Tw; j+=blockDim.x) {
			s_template[j+Tw*i] = T[j+Tw*i];
		}
	}	

	__syncthreads();
	 

	// grid-stride-loop 

	for (i = blockIdx.y * blockDim.y + threadIdx.y; i <= Ih - Th; i+=blockDim.y * gridDim.y ) {
		for (j = blockIdx.x * blockDim.x + threadIdx.x; j <= Iw - Tw; j+=blockDim.x * gridDim.x ) {
		
			// Ogni Threads esegue questa parte 
			for (k = 0; k < Th; k++) { 
				for (l = 0; l < Tw; l++) {
					temp = I[((l + j) + (k + i)*Iw)] - s_template[l + k*Tw]; // SAD 
					diff += fabsf(temp); // Valore assoluto 
				}
			}
			// Fine threads 
			differences[j + i * (Iw - Tw + 1)] = diff;
			diff = 0;
		}
	}
		
}


int main(int argc, char *argv[]){

	clock_t start = clock();

	// Durante i test ho notato che le cudamalloc() richiedevano molto tempo 
	// allora ho fatto una ricerca e suggeriscono di inserire un cudafree() all'inizio per far prepare prima il contesto Cuda ed in effetti ha funzionato
	// https://forums.developer.nvidia.com/t/cudamalloc-slow/40238/3
	cudaFree(0);

    if(argc != 4 ){
        printf("Numero di argomenti non validi!\n");
        printf("Aggiungere: immagine di origine, template e immagine di destinazione \n");
        exit(0);
    }

	
    int origine_width, origine_height, template_width, template_height, maxThreads, *index, *indexY, *d_index, *d_indexY;

    float *d_origine,  *d_template, *differences, *d_differences, *output, *d_output;
	cudaDeviceProp prop;

	
    float *origine = ReadBMP(argv[1], &origine_width, &origine_height);
	float *templat = ReadBMP(argv[2], &template_width, &template_height);
    

	// Recupero le proprietà del device
    HANDLE_ERROR( cudaGetDeviceProperties(&prop, 0));

	// Recupero il numero massimo di Threds della scheda
    maxThreads = prop.maxThreadsPerBlock; 

	// Controllo che la dimensione del template sia minore della shared memory
	if( prop.sharedMemPerBlock < sizeof(float) *template_width * template_height){
		printf("Dimensione del template troppo grande rispetto allo spazio in Shared Memory \n");
        exit(0);
	}

	// Calcolo la dimensione della matrice differences che conterrà le differenze
	int differenceW = (origine_width - template_width + 1);
	int differenceH = (origine_height - template_height + 1);

	differences = (float *)malloc( sizeof(float) * differenceW * differenceH);

	int dimThreadsPerBlock = maxThreads/32; // multiplo di 32, max 1024 (Dipende dalla scheda)
	dim3 threadsPerBlock(dimThreadsPerBlock,dimThreadsPerBlock);
	

	// Tanti blocchi quanti bastano a coprire ogni cella di differences tramite 1 thread (performance migliori)
    // Devo però allocare un numero giusto o maggiore di threads, mai minore ed essendo una divisione tra numeri interni vengono scartati i decimali, allora aggiungo maxThreads-1 
    // Es. 13 + (6-1) / 6 = 18 -> 18/6 = 3 Quindi 3 blocchi da 6 threads Totale 18
	
  	int blocks = (differenceW * differenceH + maxThreads - 1) / maxThreads;
	
	// Ogni blocco calcola un minimo nella ricerca 

	size_t output_size = blocks;

	output = (float *) malloc(output_size*sizeof(float));
    index = (int *) malloc(output_size*sizeof(int));
    indexY = (int *) malloc(output_size*sizeof(int));


	// Alloco spazio per l'immagine origine sulla GPU
	HANDLE_ERROR( cudaMalloc((void**)&d_origine, sizeof(float)*origine_width * origine_height));

	// Alloco spazio per il template sulla GPU
	HANDLE_ERROR( cudaMalloc((void**)&d_template, sizeof(float) * template_width * template_height));

    // Alloco spazio per le differenze sulla GPU
    HANDLE_ERROR( cudaMalloc((void**)&d_differences, sizeof(float) * differenceW * differenceH));

	// Alloco spazio per l'indice X sulla GPU
    HANDLE_ERROR(cudaMalloc( (void**)&d_index,output_size*sizeof(int)));
	
	// Alloco spazio per l'indice Y sulla GPU
  	HANDLE_ERROR(cudaMalloc( (void**)&d_indexY,output_size*sizeof(int)));

	// Alloco spazio per il valore minimo sulla GPU (TEST)
	HANDLE_ERROR(cudaMalloc( (void**)&d_output,output_size*sizeof(float)));

	// Copio l'immagine origine dalla CPU alla GPU
    HANDLE_ERROR(cudaMemcpy(d_origine, origine, sizeof(float)*origine_width * origine_height, cudaMemcpyHostToDevice));

    // Copio il template dalla CPU alla GPU
    HANDLE_ERROR(cudaMemcpy(d_template, templat, sizeof(float)*template_width*template_height, cudaMemcpyHostToDevice));

	
    getMatch<<<blocks,threadsPerBlock,sizeof(float) * template_width * template_height>>>(d_origine, d_template,
		origine_width, origine_height, template_width, template_height, d_differences);

  	cudaDeviceSynchronize();


	getMin<<<blocks,threadsPerBlock>>>(d_differences, differenceH, differenceW, d_output, d_index, d_indexY);
	cudaDeviceSynchronize();
	
	
	// Copio da GPU a CPU l'indice X
	HANDLE_ERROR( cudaMemcpy(index,d_index,output_size*sizeof(int),cudaMemcpyDeviceToHost));

	// Copio da GPU a CPU l'indice Y
    HANDLE_ERROR( cudaMemcpy(indexY,d_indexY,output_size*sizeof(int),cudaMemcpyDeviceToHost));

	// Copio da GPU a CPU il valore minimo (TEST)
    HANDLE_ERROR( cudaMemcpy(output,d_output,output_size*sizeof(float),cudaMemcpyDeviceToHost));
	
	
	// Per Test
	//printf("minimo: %f\n", output[0]);
	//printf("X: %i\n", index[0]);
	//printf("Y: %i\n", indexY[0]);
	
	
	free(differences);
	cudaFree(d_origine);
	cudaFree(d_template);
	cudaFree(d_differences);
	cudaFree(d_output);
	cudaFree(d_index);
	cudaFree(d_indexY);

	int x1, x2, y1, y2;
    x1 = index[0];
    x2 = index[0] + template_width - 1;
    y1 = indexY[0];
    y2 = indexY[0] + template_height - 1;

    MarkAndSave(argv[1], x1, y1, x2, y2, argv[3]);
    printf("Percorso immagine risultante: %s\n", argv[3]);

	clock_t end = clock();
	printf("Tempo di esecuzione =  %f secondi \n", ((double)(end - start)) / CLOCKS_PER_SEC);

}