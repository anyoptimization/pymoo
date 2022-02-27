#ifndef _WFG_H_
#define _WFG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double OBJECTIVE;

typedef struct
{
	OBJECTIVE *objectives;
      struct avl_node_t * tnode;
} POINT;

typedef struct
{
	int nPoints;
	int n;
	POINT *points;
} FRONT;

typedef struct
{
	int nFronts;
	FRONT *fronts;
} FILECONTENTS;

FILECONTENTS *readFile(char[]);

extern void printContents(FILECONTENTS *);

#endif