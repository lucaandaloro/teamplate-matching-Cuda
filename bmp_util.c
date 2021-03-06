#include "includes/bmp_util.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include "includes/bmp.h"
#endif

float *ReadBMP(const char *bmpName, int *width, int *height) {
  FILE *fp;
  uchar *img_raw;
  float *image;
  int bmpwidth, bmpheight, linebyte, npixels, i, j;
  if ((fp = fopen(bmpName, "rb")) == NULL) {
    printf("Errore nell'apertura dell'immagine.\n");
    return 0;
  }

  if (fseek(fp, sizeof(BITMAPFILEHEADER), 0)) {
    printf("Errore nel file header.\n");
    return 0;
  }

  BITMAPINFOHEADER bmpInfoHeader;
  fread(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
  bmpwidth = bmpInfoHeader.biWidth;
  bmpheight = bmpInfoHeader.biHeight;
  npixels = bmpwidth * bmpheight;
  linebyte = (bmpwidth * 24 / 8 + 3) / 4 * 4;

  img_raw = (uchar *)malloc(linebyte * bmpheight);
  fread(img_raw, linebyte * bmpheight, 1, fp); // Leggiamo l'immagine aperta con fopen in precedenza

  image = (float *)malloc(sizeof(float) * npixels);
  for (i = 0; i < bmpheight; i++)
    for (j = 0; j < bmpwidth; j++)
      image[i * bmpwidth + j] = 0.11 * img_raw[i * linebyte + j * 3] + // R
                                0.59 * img_raw[i * linebyte + j * 3 + 1] + // G
                                0.30 * img_raw[i * linebyte + j * 3 + 2]; // B
  *width = bmpwidth;
  *height = bmpheight;

  free(img_raw);
  fclose(fp);
  return image;
}

void MarkAndSave(const char *bmpName, int X1, int Y1, int X2, int Y2,
                 const char *outputBmpName) {
  FILE *fp;
  uchar *img_raw;
  // float *image;
  BITMAPFILEHEADER bmpFileHeader;
  BITMAPINFOHEADER bmpInfoHeader;
  int bmpwidth, bmpheight, linebyte;  //, npixels;
  if ((fp = fopen(bmpName, "rb")) == NULL) {
    printf("Errore nell'apertura dell'immagine originale.\n");
    return;
  }

  fread(&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
  fread(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
  bmpwidth = bmpInfoHeader.biWidth;
  bmpheight = bmpInfoHeader.biHeight;
  // npixels = bmpwidth * bmpheight;
  linebyte = (bmpwidth * 24 / 8 + 3) / 4 * 4;

  img_raw = (uchar *)malloc(linebyte * bmpheight);
  fread(img_raw, linebyte * bmpheight, 1, fp);
  fclose(fp);

  if (X1 < 0 || X2 >= bmpwidth || Y1 < 0 || Y2 >= bmpheight) {
    printf("Posizione non valida!\n");
    return;
  }
  int i;

  // Disegniamo il rettangolo
  for (i = X1; i <= X2; i++) {
    img_raw[Y1 * linebyte + i * 3] = 0;
    img_raw[Y1 * linebyte + i * 3 + 1] = 0;
    img_raw[Y1 * linebyte + i * 3 + 2] = 255;
    img_raw[Y2 * linebyte + i * 3] = 0;
    img_raw[Y2 * linebyte + i * 3 + 1] = 0;
    img_raw[Y2 * linebyte + i * 3 + 2] = 255;
  }
  for (i = Y1 + 1; i < Y2; i++) {
    img_raw[i * linebyte + X1 * 3] = 0;
    img_raw[i * linebyte + X1 * 3 + 1] = 0;
    img_raw[i * linebyte + X1 * 3 + 2] = 255;
    img_raw[i * linebyte + X2 * 3] = 0;
    img_raw[i * linebyte + X2 * 3 + 1] = 0;
    img_raw[i * linebyte + X2 * 3 + 2] = 255;
  }

  if ((fp = fopen(outputBmpName, "wb")) == NULL) {
    printf("Errore nell'apertura dell'immagine di output.\n");
    return;
  }
  fwrite(&bmpFileHeader, sizeof(BITMAPFILEHEADER), 1, fp);
  fwrite(&bmpInfoHeader, sizeof(BITMAPINFOHEADER), 1, fp);
  fwrite(img_raw, linebyte * bmpheight, 1, fp);

  free(img_raw);
  fclose(fp);
}