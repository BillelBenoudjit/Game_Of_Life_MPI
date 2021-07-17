#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"
#include <time.h>

#define MAX_ALIVE_CELLS 4194304

//--- Variables to calculate execution time --- 
float execTime; 
clock_t start, end; 

struct coord {
  int x;
  int y;
};

struct alive_tab {
  int length;
  struct coord alive[4194304];
};

struct the_board {
  int is_alive;
  int point;
};

int main(int argc, char **argv) {
  //--- initialize MPI ---
  MPI_Status status;   // required variable for receive routines
  int noTasks = 2;
  int rank = 0;
  
  MPI_Init( &argc, &argv );
  
  //--- get number of tasks, and make sure it's 2 ---
  MPI_Comm_size( MPI_COMM_WORLD, &noTasks );
  if ( noTasks != 2 ) {
    printf( "Number of Processes/Tasks must be 2.  Number = %d\n\n", noTasks );
    MPI_Finalize();
    return 1;
  }

  //--- get rank ---
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int mpiRoot = 0;
  int nRows, nCols, nIters;
  
  if (rank == mpiRoot) {
    nRows = 2048;
    nCols = 2048;
    nIters = 200;
  }
  
  int cpt = 0;
  int x, y;
  
  // each processor calls MPI_Bcast, data is broadcast from root processor and ends up in everyone value variable
  // root process uses MPI_Bcast to broadcast the value, each other process uses MPI_Bcast to receive the broadcast value
  MPI_Bcast(&nRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nIters, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int nRowsLocal = nRows / noTasks;
  int nRowsLocalWithGhost = nRowsLocal + 2;
  int nColsWithGhost = nCols + 2;
  int i = 0;
  
  static struct the_board board[2048][2048];
  static struct the_board even_board[1026][2050];
  static struct the_board odd_board[1026][2050];
  
  static struct alive_tab AliveTab;
  AliveTab.length = 0;
  
  for (int iRow = 0; iRow < nRowsLocalWithGhost; ++iRow) {
    for (int iCol = 0; iCol < nColsWithGhost; ++iCol) {
      even_board[iRow][iCol].is_alive = 0;
      even_board[iRow][iCol].point = 0;
      odd_board[iRow][iCol].is_alive = 0;
      odd_board[iRow][iCol].point = 0;
    }
  }
  
  //--- create a type for struct the_board ---
  const int nitems = 2;
  int blocklengths[2] = {1,1};
  MPI_Datatype types[2] = {MPI_INT, MPI_INT};
  MPI_Datatype mpi_board_type;
  MPI_Aint offsets[2];
  
  offsets[0] = offsetof(struct the_board, is_alive);
  offsets[1] = offsetof(struct the_board, point);
  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_board_type);
  MPI_Type_commit(&mpi_board_type);

  //--- init present board ---
  FILE *fptr;
  int num;

  if ((fptr = fopen("./data", "r")) == NULL) {
    printf("error opening file");
    exit(1);
  }
  
  for (int iRow = 0; iRow < nRows; ++iRow) {
    for (int iCol = 0; iCol < nCols; ++iCol) {
      fscanf(fptr, "%d", &num);
      //printf("%d\n", num);
      board[iRow][iCol].is_alive = num;
      if (num == 1) {
        if (AliveTab.length < MAX_ALIVE_CELLS) {
          if (iRow <= nRowsLocalWithGhost) {
      	    AliveTab.alive[AliveTab.length].x = iRow + 1;
      	  } else {
      	    AliveTab.alive[AliveTab.length].x = iRow;
      	  }
          AliveTab.alive[AliveTab.length].y = iCol + 1;
          AliveTab.length++;
        }
      }
      board[iRow][iCol].point = 0;  
    }
  }
  
  fclose(fptr);

  //--- init even & odd boards ---
  if (rank == 0) {
    for (int iRow = 1; iRow <= nRowsLocal; ++iRow) {
      for (int iCol = 1; iCol <= nCols; ++iCol) {
        even_board[iRow][iCol].is_alive = board[iRow-1][iCol-1].is_alive;
      	even_board[iRow][iCol].point = board[iRow-1][iCol-1].point;
      }
    }
  } else {
    for (int iRow = 1; iRow <= nRowsLocal; ++iRow) {
      for (int iCol = 1; iCol <= nCols; ++iCol) {
        odd_board[iRow][iCol].is_alive = board[iRow + nRowsLocal - 1][iCol - 1].is_alive;
      	odd_board[iRow][iCol].point = board[iRow + nRowsLocal - 1][iCol - 1].point;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  if (rank == 0) {
    start = clock(); 
  }
  
  //--- iterate ---
  for (int iIter = 0; iIter < nIters; ++iIter) {
    int send;
    int recv;
    
    if (rank == 0) {
      //printf("-------- Iteration %d --------\n", iIter);
      
      //--- ghost columns ---
      for (int iRow = 1; iRow < nRowsLocalWithGhost; ++iRow) {
        even_board[iRow][0].is_alive = board[iRow-1][nCols-1].is_alive;
        even_board[iRow][0].point = board[iRow-1][nCols-1].point;
      
        even_board[iRow][nCols + 1].is_alive = board[iRow-1][0].is_alive;
        even_board[iRow][nCols + 1].point = board[iRow-1][0].point;
      }
      
      //--- Send ghost rows to rank 1 --- 
      send = MPI_Send(&even_board[1][0], nColsWithGhost, mpi_board_type, 1, 1, MPI_COMM_WORLD);
      send = MPI_Send(&even_board[nRowsLocal][0], nColsWithGhost, mpi_board_type, 1, 1, MPI_COMM_WORLD);
      
      // -- Recieve ghost rows from rank 1 ---
      recv = MPI_Recv(&even_board[0][0], nColsWithGhost, mpi_board_type, 1, 1, MPI_COMM_WORLD, &status);
      recv = MPI_Recv(&even_board[nRowsLocal + 1][0], nColsWithGhost, mpi_board_type, 1, 1, MPI_COMM_WORLD, &status);
    
    } else {
      //--- ghost columns ---
      for (int iRow = 1; iRow <= nRowsLocal; ++iRow) {
          odd_board[iRow][0].is_alive = board[iRow-1+nRowsLocal][nCols-1].is_alive;
          odd_board[iRow][0].point = board[iRow-1+nRowsLocal][nCols-1].point;
          
          odd_board[iRow][nCols + 1].is_alive = board[iRow-1+nRowsLocal][0].is_alive;
          odd_board[iRow][nCols + 1].point = board[iRow-1+nRowsLocal][0].point;
      }
      
      // -- Recieve ghost rows from rank 0 ---
      recv = MPI_Recv(&odd_board[nRowsLocal + 1][0], nColsWithGhost, mpi_board_type, 0, 1, MPI_COMM_WORLD, &status);
      recv = MPI_Recv(&odd_board[0][0], nColsWithGhost, mpi_board_type, 0, 1, MPI_COMM_WORLD, &status);
      
      //--- Send ghost rows to rank 0 ---
      send = MPI_Send(&odd_board[nRowsLocal][0], nColsWithGhost, mpi_board_type, 0, 1, MPI_COMM_WORLD);
      send = MPI_Send(&odd_board[1][0], nColsWithGhost, mpi_board_type, 0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    //--- Update even & odd boards ---
    for (int i = 0; i < AliveTab.length; i++) {
      x = AliveTab.alive[i].x;
      y = AliveTab.alive[i].y;
      
      if (rank == 0) {
        if (x <= nRowsLocal) {
          even_board[x][y + 1].point = even_board[x][y + 1].point + 1;
          even_board[x][y - 1].point = even_board[x][y - 1].point + 1;

          even_board[x + 1][y - 1].point = even_board[x + 1][y - 1].point + 1;
          even_board[x + 1][y].point = even_board[x + 1][y].point + 1;
          even_board[x + 1][y + 1].point = even_board[x + 1][y + 1].point + 1;

          even_board[x - 1][y - 1].point = even_board[x - 1][y - 1].point + 1;
          even_board[x - 1][y].point = even_board[x - 1][y].point + 1;
          even_board[x - 1][y + 1].point = even_board[x - 1][y + 1].point + 1;
        }
      } else {
        if (x > nRowsLocal) {
          x = (x % nRowsLocal) + 1;
          odd_board[x][y + 1].point = odd_board[x][y + 1].point + 1;
          odd_board[x][y - 1].point = odd_board[x][y - 1].point + 1;

          odd_board[x + 1][y - 1].point = odd_board[x + 1][y - 1].point + 1;
          odd_board[x + 1][y].point = odd_board[x + 1][y].point + 1;
          odd_board[x + 1][y + 1].point = odd_board[x + 1][y + 1].point + 1;

          odd_board[x - 1][y - 1].point = odd_board[x - 1][y - 1].point + 1;
          odd_board[x - 1][y].point = odd_board[x - 1][y].point + 1;
          odd_board[x - 1][y + 1].point = odd_board[x - 1][y + 1].point + 1;
        }  
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    AliveTab.length = 0; // We are going to reconstruct a new Table of living cells
    for (int i = 1; i <= nRowsLocal; i++) {
      for (int j = 1; j <= nCols; j++) {
      	if (rank == 0) {
          if (even_board[i][j].is_alive == 1) {
            if ((even_board[i][j].point > 3) || (even_board[i][j].point < 2)) {
              even_board[i][j].is_alive = 0; // Die because of overload or less neighbor
            } else {
              if (AliveTab.length < MAX_ALIVE_CELLS) {
                AliveTab.alive[AliveTab.length].x = i;
                AliveTab.alive[AliveTab.length].y = j;
                AliveTab.length++;
              }
            }
          } else {
            if (even_board[i][j].point == 3) {
              if (AliveTab.length < MAX_ALIVE_CELLS) {
                even_board[i][j].is_alive = 1; // Birth of the cell
                AliveTab.alive[AliveTab.length].x = i;
                AliveTab.alive[AliveTab.length].y = j;
                AliveTab.length++;
              }
            }
          }
          even_board[i][j].point = 0;
        } else {
          if (odd_board[i][j].is_alive == 1) {
            if ((odd_board[i][j].point > 3) || (odd_board[i][j].point < 2)) {
              odd_board[i][j].is_alive = 0; // Die because of overload or less neighbor
            } else {
              if (AliveTab.length < MAX_ALIVE_CELLS) {
                AliveTab.alive[AliveTab.length].x = i + nRowsLocal - 1;
                AliveTab.alive[AliveTab.length].y = j;
                AliveTab.length++;
              }
            }
          } else {
            if (odd_board[i][j].point == 3) {
              if (AliveTab.length < MAX_ALIVE_CELLS) {
                odd_board[i][j].is_alive = 1; // Birth of the cell
                AliveTab.alive[AliveTab.length].x = i + nRowsLocal - 1;
                AliveTab.alive[AliveTab.length].y = j;
                AliveTab.length++;
              }
            }
          }
          odd_board[i][j].point = 0;
        }
      }
    }
    
    //--- Update board from even & odd boards ---    
    if (rank == 0) {
      for (int iRow = 1; iRow <= nRowsLocal; ++iRow) {
        for (int iCol = 1; iCol <= nCols; ++iCol) {
          board[iRow - 1][iCol - 1].is_alive = even_board[iRow][iCol].is_alive;
          even_board[iRow][iCol].point = 0;
          board[iRow - 1][iCol - 1].point = 0;
        }
      }
    } else {
      for (int iRow = 1; iRow <= nRowsLocal; ++iRow) {
        for (int iCol = 1; iCol <= nCols; ++iCol) {
          board[iRow + nRowsLocal - 1][iCol - 1].is_alive = odd_board[iRow][iCol].is_alive;
          odd_board[iRow][iCol].point = 0;
          board[iRow + nRowsLocal - 1][iCol - 1].point = 0;
        }
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    //--- display board ---
    /*if (rank == 0) {
      printf("\n--- Rank %d ---\n", rank);
      for (int iRow = 0; iRow < nRows; ++iRow) {
        for (int iCol = 0; iCol < nCols; ++iCol) {
          if (board[iRow][iCol].is_alive == 1) {
            printf("  %d  ", board[iRow][iCol].is_alive);
          } else {
            printf("  %d  ", board[iRow][iCol].is_alive);
          }
          if (iCol == nCols - 1) {
            printf("\n\n");
          }
        }
      }
    }
    if (rank == 1) {
      printf("\n--- Rank %d ---\n", rank);
      for (int iRow = 0; iRow < nRows; ++iRow) {
        for (int iCol = 0; iCol < nCols; ++iCol) {
          if (board[iRow][iCol].is_alive == 1) {
            printf("  %d  ", board[iRow][iCol].is_alive);
          } else {
            printf("  %d  ", board[iRow][iCol].is_alive);
          }
          if (iCol == nCols - 1) {
            printf("\n\n");
          }
        }
      }
    }*/
  }
  
  //printf("Rank %d Finalize\n", rank);
  
  //--- get time --- 
  if (rank == 0) {
    end = clock(); 
    execTime = (float)(end - start) / CLOCKS_PER_SEC; 
    printf("\n------------------------------------------------\n");
    printf("Exectution Time with MPI - Game of life = %f s\n", execTime);
    printf("For %ld iterations & board size = %ld*%ld.\n", nIters, nRows, nCols);
    printf("------------------------------------------------\n");	
  }
  
  MPI_Finalize();
  
  return 0;
}
