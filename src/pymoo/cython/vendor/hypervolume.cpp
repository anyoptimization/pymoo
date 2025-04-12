#include "hypervolume.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>



/* Auxiliary functions and variables for the Overmars-Yap algorithm. */
static unsigned NO_OBJECTIVES;
static double	SQRT_NO_DATA_POINTS;

int compare( const void * a, const void * b ) {
	double * x = (double*) a;
	double * y = (double*) b;

	if( x[NO_OBJECTIVES-1] == y[NO_OBJECTIVES-1] )
		return( 0 );

	if( x[NO_OBJECTIVES-1] < y[NO_OBJECTIVES-1] )
		return( -1 );

	// if( x[NO_OBJECTIVES-1] > y[NO_OBJECTIVES-1] )

	return( 1 );
}

int 	covers				( double * cuboid, 		double * regionLow						);
int 	partCovers			( double * cuboid, 		double * regionUp						);
int 	containsBoundary	( double * cub, 		double * regLow, 	int split			);
double 	getMeasure			( double * regionLow, 	double * regionUp						);
int 	isPile				( double * cuboid, 		double * regionLow, double * regionUp	);
int 	binaryToInt			( int * bs );
void 	intToBinary			( int i, 				int * result							);
double 	computeTrellis		( double * regLow, 		double * regUp, 	double * trellis	);
double 	getMedian			( double * bounds, 		int length								);

double 	stream				( double * regionLow, 	double * regionUp, 	double * points,
							  unsigned noPoints, 	int split, 			double cover		);

double overmars_yap( double * points, double * referencePoint, unsigned noObjectives, unsigned noPoints ) {
	NO_OBJECTIVES 		= noObjectives;
	SQRT_NO_DATA_POINTS = sqrt( (double)noPoints );

	double * regLow = new double[NO_OBJECTIVES];
	std::fill( regLow, regLow + NO_OBJECTIVES, std::numeric_limits<double>::max() );

	// Calculate Bounding Box
	double * p = points;
	for( unsigned i = 0; i < noPoints; i++ ) {
		for( unsigned j = 0; j < NO_OBJECTIVES; j++ ) {
			regLow[j] = std::min( regLow[j], *p );

			++p;
		}
	}

	double d = stream( regLow, referencePoint, points, noPoints, 0, referencePoint[NO_OBJECTIVES-1] );

	delete [] regLow;
	return( d );
}

int covers(double * cuboid, double * regionLow)
{
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		if (cuboid[i] > regionLow[i])
			return (0);
	}
	return (1);
}

int partCovers( double * cuboid, double * regionUp ) {
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++) {
		if (cuboid[i] >= regionUp[i])
			return (0);
	}
	return (1);
}

int containsBoundary(double * cub, double * regLow, int split) {
	if (regLow[split] >= cub[split]) {
		return -1;
	} else {
		int j;
		for (j = 0; j < split; j++) {
			if (regLow[j] < cub[j]) {
				return 1;
			}
		}
	}
	return 0;
}

double getMeasure(double * regionLow, double * regionUp)
{
	double volume = 1.0;
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		volume *= (regionUp[i] - regionLow[i]);
	}

	return (volume);
}

int isPile(double * cuboid, double * regionLow, double * regionUp)
{
	unsigned pile = NO_OBJECTIVES;
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		if (cuboid[i] > regionLow[i])
		{
			if (pile != NO_OBJECTIVES)
			{
				return (-1);
			}

			pile = i;
		}
	}

	return (pile);
}

int binaryToInt(int * bs)
{
	int result = 0;
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		result += bs[i] * (int) pow(2.0, (double)i);
	}

	return (result);
}

void intToBinary(int i, int * result)
{
	unsigned j;
	for (j = 0; j < NO_OBJECTIVES - 1; j++)
		result[j] = 0;

	int rest = i;
	int idx = 0;

	while (rest != 0)
	{
		result[idx] = (rest % 2);

		rest = rest / 2;
		idx++;
	}
}

double computeTrellis(double * regLow, double * regUp, double * trellis)
{
	unsigned i, j;
	int * bs = (int*)malloc((NO_OBJECTIVES - 1) * sizeof(int));
	for (i = 0; i < NO_OBJECTIVES - 1; i++) bs[i] = 1;

	double result = 0;

	int noSummands = binaryToInt(bs);
	int oneCounter; double summand;

	for (i = 1; i <= (unsigned)noSummands; i++)
	{
		summand = 1;
		intToBinary(i, bs);
		oneCounter = 0;

		for (j = 0; j < NO_OBJECTIVES - 1; j++)
		{
			if (bs[j] == 1)
			{
				summand *= regUp[j] - trellis[j];
				oneCounter++;
			}
			else
				summand *= regUp[j] - regLow[j];
		}

		if (oneCounter % 2 == 0)
			result -= summand ;
		else
			result += summand;
	}

	free(bs);

	return(result);
}

int double_compare(const void *p1, const void *p2)
{
	double i = *((double *)p1);
	double j = *((double *)p2);

	if (i > j)
		return (1);
	if (i < j)
		return (-1);
	return (0);
}

double getMedian(double * bounds, int length)
{
	if (length == 1)
	{
		return bounds[0];
	}
	else if (length == 2)
	{
		return bounds[1];
	}

	qsort(bounds, length, sizeof(double), double_compare);

	return(length % 2 == 1 ? bounds[length / 2] : (bounds[length / 2] + bounds[length / 2 + 1]) / 2);
}

double stream(double * regionLow,
			  double * regionUp,
			  double * points,
			  unsigned noPoints,
			  int split,
			  double cover ) {
        
    using namespace std;
    
	double coverOld;
	coverOld = cover;
	int coverIndex = 0;
	int c;

	double result = 0;

	double dMeasure = getMeasure(regionLow, regionUp);
	while (cover == coverOld && coverIndex < (double)noPoints)
	{
		if (covers(points + (coverIndex * NO_OBJECTIVES), regionLow))
		{
			cover = points[coverIndex * NO_OBJECTIVES + NO_OBJECTIVES - 1];
			result += dMeasure * (coverOld - cover);
		}
		else
			coverIndex++;
	}

	for (c = coverIndex; c > 0; c--) {
		if (points[(c - 1) * NO_OBJECTIVES + NO_OBJECTIVES - 1] == cover) {
			coverIndex--;
		}
	}

	if (coverIndex == 0)
	{
		return (result);
	}

	int allPiles = 1; int i;

	int  * piles = (int*)malloc(coverIndex * sizeof(int));
	for (i = 0; i < coverIndex; i++)
	{
		piles[i] = isPile(points + i * NO_OBJECTIVES, regionLow, regionUp);
		if (piles[i] == -1)
		{
			allPiles = 0;
			break;
		}
	}

	if (allPiles)
	{
		double * trellis = (double*)malloc((NO_OBJECTIVES - 1) * sizeof(double));
		for (c = 0; c < (int)NO_OBJECTIVES - 1; c++)
		{
			trellis[c] = regionUp[c];
		}

		double current = 0.0;
		double next = 0.0;
		i = 0;
		do
		{
			current = points[i * NO_OBJECTIVES + NO_OBJECTIVES - 1];
			do
			{
				if ( points[i * NO_OBJECTIVES + piles[i]] < trellis[piles[i]])
				{
					trellis[piles[i]] = points[i * NO_OBJECTIVES + piles[i]];
				}
				i++;
				if (i < coverIndex)
				{
					next = points[i * NO_OBJECTIVES + NO_OBJECTIVES - 1];
				}
				else
				{
					next = cover;
					break;
				}

			}
			while (next == current);
			result += computeTrellis(regionLow, regionUp, trellis)
					  * (next - current);
		}
		while (next != cover);
		free(trellis);
	}
	else
	{
		double bound = -1.0;
		double * boundaries = (double*) malloc(coverIndex * sizeof(double));
		unsigned boundIdx = 0;
		double * noBoundaries = (double*)malloc(coverIndex * sizeof(double));
		unsigned noBoundIdx = 0;

		do
		{
			for (i = 0; i < coverIndex; i++)
			{
				int contained = containsBoundary(points + i * NO_OBJECTIVES, regionLow, split);
				if (contained == 1)
				{
					boundaries[boundIdx] = points[i * NO_OBJECTIVES + split];
					boundIdx++;
				}
				else if (contained == 0)
				{
					noBoundaries[noBoundIdx] = points[i * NO_OBJECTIVES + split];
					noBoundIdx++;
				}
			}

			if (boundIdx > 0)
			{
				bound = getMedian(boundaries, boundIdx);
			}
			else if (noBoundIdx > SQRT_NO_DATA_POINTS)
			{
				bound = getMedian(noBoundaries, noBoundIdx);
			}
			else
			{
				split++;
			}
		}
		while (bound == -1.0);

		free(boundaries); free(noBoundaries);

		double * pointsChild = new double[coverIndex * NO_OBJECTIVES];//(doublep*)malloc(coverIndex * sizeof(doublep*));
		int pointsChildIdx = 0;

		double * regionUpC = (double*)malloc(NO_OBJECTIVES * sizeof(double));
		memcpy(regionUpC, regionUp, NO_OBJECTIVES * sizeof(double));
		regionUpC[split] = bound;

		for (i = 0; i < coverIndex; i++)
		{
			if (partCovers(points + i * NO_OBJECTIVES, regionUpC))
			{
				std::copy( points + i*NO_OBJECTIVES, points + i*NO_OBJECTIVES + NO_OBJECTIVES, pointsChild + pointsChildIdx*NO_OBJECTIVES );
				//pointsChild[pointsChildIdx] = points[i * NO_OBJECTIVES];
				pointsChildIdx++;
			}
		}

		if (pointsChildIdx > 0)
		{
			result += stream(regionLow, regionUpC, pointsChild, pointsChildIdx, split, cover);
		}

		pointsChildIdx = 0;

		double * regionLowC = (double*)malloc(NO_OBJECTIVES * sizeof(double));
		memcpy(regionLowC, regionLow, NO_OBJECTIVES * sizeof(double));
		regionLowC[split] = bound;
		for (i = 0; i < coverIndex; i++)
		{
			if (partCovers(points + i * NO_OBJECTIVES, regionUp))
			{
				// pointsChild[pointsChildIdx] = points[i];
				std::copy( points + i*NO_OBJECTIVES, points + i*NO_OBJECTIVES + NO_OBJECTIVES, pointsChild + pointsChildIdx*NO_OBJECTIVES );
				pointsChildIdx++;
			}
		}
		if (pointsChildIdx > 0)
		{
			result += stream(regionLowC, regionUp, pointsChild, pointsChildIdx, split, cover);
		}

		free(regionUpC);
		free(regionLowC);
		delete [] pointsChild;
	}

	free(piles);

	return (result);
}

/************************************************************************
Relevant literature:

 [1]  C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An
      improved dimension-sweep algorithm for the hypervolume
      indicator. In IEEE Congress on Evolutionary Computation,
      pages 1157-1163, Vancouver, Canada, July 2006.

 [2]  L. Paquete, C. M. Fonseca and M. Lopez-Ibanez. An optimal
      algorithm for a special case of Klee's measure problem in three
      dimensions. Technical Report CSI-RT-I-01/2006, CSI, Universidade
      do Algarve, 2006.

*************************************************************************/

/*****************************************************************************

    avl.c & avl.h - Source code for the AVL-tree library.

    Copyright (C) 1998  Michael H. Buselli <cosine@cosine.org>
    Copyright (C) 2000-2002  Wessel Dankers <wsl@nl.linux.org>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA

    Augmented AVL-tree. Original by Michael H. Buselli <cosine@cosine.org>.

    Modified by Wessel Dankers <wsl@nl.linux.org> to add a bunch of bloat to
    the sourcecode, change the interface and squash a few bugs.
    Mail him if you find new bugs.

*****************************************************************************/

#define VARIANT 3

#if !defined(AVL_DEPTH) && !defined(AVL_COUNT)
#define AVL_DEPTH
#define AVL_COUNT
#endif

static const int stop_dimension = 1;
static std::vector<void*> allocated_pointers;

static void * ini_malloc( unsigned int nrBytes ) {
	void * result = malloc( nrBytes );
	if( result != 0 )
		allocated_pointers.push_back( result );

	return( result );
}

static void ini_free( void * p, bool reallyFree = false ) {
	std::vector<void*>::iterator it = std::find( allocated_pointers.begin(), allocated_pointers.end(), p );
	if( it == allocated_pointers.end() )
		return;
	allocated_pointers.erase( it );
	if( reallyFree )
		free( p );
}

/* User supplied function to compare two items like strcmp() does.
 * For example: cmp(a,b) will return:
 *   -1  if a < b
 *    0  if a = b
 *    1  if a > b
 */
typedef int (*avl_compare_t)(const void *, const void *);

/* User supplied function to delete an item when a node is free()d.
 * If NULL, the item is not free()d.
 */
typedef void (*avl_freeitem_t)(void *);

typedef struct avl_node_t {
	struct avl_node_t *next;
	struct avl_node_t *prev;
	struct avl_node_t *parent;
	struct avl_node_t *left;
	struct avl_node_t *right;
	void *item;
#ifdef AVL_COUNT
	unsigned int count;
#endif
#ifdef AVL_DEPTH
	unsigned char depth;
#endif
} avl_node_t;

typedef struct avl_tree_t {
	avl_node_t *head;
	avl_node_t *tail;
	avl_node_t *top;
	avl_compare_t cmp;
	avl_freeitem_t freeitem;
} avl_tree_t;

/* Initializes a new tree for elements that will be ordered using
 * the supplied strcmp()-like function.
 * Returns the value of avltree (even if it's NULL).
 * O(1) */
extern avl_tree_t *avl_init_tree(avl_tree_t *avltree, avl_compare_t, avl_freeitem_t);

/* Allocates and initializes a new tree for elements that will be
 * ordered using the supplied strcmp()-like function.
 * Returns NULL if memory could not be allocated.
 * O(1) */
extern avl_tree_t *avl_alloc_tree(avl_compare_t, avl_freeitem_t);

/* Frees the entire tree efficiently. Nodes will be free()d.
 * If the tree's freeitem is not NULL it will be invoked on every item.
 * O(n) */
extern void avl_free_tree(avl_tree_t *);

/* Reinitializes the tree structure for reuse. Nothing is free()d.
 * Compare and freeitem functions are left alone.
 * O(1) */
extern void avl_clear_tree(avl_tree_t *);

/* Free()s all nodes in the tree but leaves the tree itself.
 * If the tree's freeitem is not NULL it will be invoked on every item.
 * O(n) */
extern void avl_free_nodes(avl_tree_t *);

/* Initializes memory for use as a node. Returns NULL if avlnode is NULL.
 * O(1) */
extern avl_node_t *avl_init_node(avl_node_t *avlnode, void *item);

/* Insert an item into the tree and return the new node.
 * Returns NULL and sets errno if memory for the new node could not be
 * allocated or if the node is already in the tree (EEXIST).
 * O(lg n) */
extern avl_node_t *avl_insert(avl_tree_t *, void *item);

/* Insert a node into the tree and return it.
 * Returns NULL if the node is already in the tree.
 * O(lg n) */
extern avl_node_t *avl_insert_node(avl_tree_t *, avl_node_t *);

/* Insert a node in an empty tree. If avlnode is NULL, the tree will be
 * cleared and ready for re-use.
 * If the tree is not empty, the old nodes are left dangling.
 * O(1) */
extern avl_node_t *avl_insert_top(avl_tree_t *, avl_node_t *avlnode);

/* Insert a node before another node. Returns the new node.
 * If old is NULL, the item is appended to the tree.
 * O(lg n) */
extern avl_node_t *avl_insert_before(avl_tree_t *, avl_node_t *old, avl_node_t * newNode);

/* Insert a node after another node. Returns the new node.
 * If old is NULL, the item is prepended to the tree.
 * O(lg n) */
extern avl_node_t *avl_insert_after(avl_tree_t *, avl_node_t *old, avl_node_t *newNode);

/* Deletes a node from the tree. Returns immediately if the node is NULL.
 * The item will not be free()d regardless of the tree's freeitem handler.
 * This function comes in handy if you need to update the search key.
 * O(lg n) */
extern void avl_unlink_node(avl_tree_t *, avl_node_t *);

/* Deletes a node from the tree. Returns immediately if the node is NULL.
 * If the tree's freeitem is not NULL, it is invoked on the item.
 * If it is, returns the item.
 * O(lg n) */
extern void *avl_delete_node(avl_tree_t *, avl_node_t *);

/* Searches for an item in the tree and deletes it if found.
 * If the tree's freeitem is not NULL, it is invoked on the item.
 * If it is, returns the item.
 * O(lg n) */
extern void *avl_delete(avl_tree_t *, const void *item);

/* If exactly one node is moved in memory, this will fix the pointers
 * in the tree that refer to it. It must be an exact shallow copy.
 * Returns the pointer to the old position.
 * O(1) */
extern avl_node_t *avl_fixup_node(avl_tree_t *, avl_node_t *newNode);

/* Searches for a node with the key closest (or equal) to the given item.
 * If avlnode is not NULL, *avlnode will be set to the node found or NULL
 * if the tree is empty. Return values:
 *   -1  if the returned node is smaller
 *    0  if the returned node is equal or if the tree is empty
 *    1  if the returned node is greater
 * O(lg n) */
extern int avl_search_closest(const avl_tree_t *, const void *item, avl_node_t **avlnode);

/* Searches for the item in the tree and returns a matching node if found
 * or NULL if not.
 * O(lg n) */
extern avl_node_t *avl_search(const avl_tree_t *, const void *item);

#ifdef AVL_COUNT
/* Returns the number of nodes in the tree.
 * O(1) */
extern unsigned int avl_count(const avl_tree_t *);

/* Searches a node by its rank in the list. Counting starts at 0.
 * Returns NULL if the index exceeds the number of nodes in the tree.
 * O(lg n) */
extern avl_node_t *avl_at(const avl_tree_t *, unsigned int);

/* Returns the rank of a node in the list. Counting starts at 0.
 * O(lg n) */
extern unsigned int avl_index(const avl_node_t *);
#endif

static void avl_rebalance(avl_tree_t *, avl_node_t *);

#ifdef AVL_COUNT
#define NODE_COUNT(n)  ((n) ? (n)->count : 0)
#define L_COUNT(n)     (NODE_COUNT((n)->left))
#define R_COUNT(n)     (NODE_COUNT((n)->right))
#define CALC_COUNT(n)  (L_COUNT(n) + R_COUNT(n) + 1)
#endif

#ifdef AVL_DEPTH
#define NODE_DEPTH(n)  ((n) ? (n)->depth : 0)
#define L_DEPTH(n)     (NODE_DEPTH((n)->left))
#define R_DEPTH(n)     (NODE_DEPTH((n)->right))
#define CALC_DEPTH(n)  ((L_DEPTH(n)>R_DEPTH(n)?L_DEPTH(n):R_DEPTH(n)) + 1)
#endif

#ifndef AVL_DEPTH
/* Also known as ffs() (from BSD) */
static int lg(unsigned int u) {
	int r = 1;
	if(!u) return 0;
	if(u & 0xffff0000) { u >>= 16; r += 16; }
	if(u & 0x0000ff00) { u >>= 8; r += 8; }
	if(u & 0x000000f0) { u >>= 4; r += 4; }
	if(u & 0x0000000c) { u >>= 2; r += 2; }
	if(u & 0x00000002) r++;
	return r;
}
#endif

static int avl_check_balance(avl_node_t *avlnode) {
#ifdef AVL_DEPTH
	int d;
	d = R_DEPTH(avlnode) - L_DEPTH(avlnode);
	return d<-1?-1:d>1?1:0;
#else
/*	int d;
 *	d = lg(R_COUNT(avlnode)) - lg(L_COUNT(avlnode));
 *	d = d<-1?-1:d>1?1:0;
 */
#ifdef AVL_COUNT
	int pl, r;

	pl = lg(L_COUNT(avlnode));
	r = R_COUNT(avlnode);

	if(r>>pl+1)
		return 1;
	if(pl<2 || r>>pl-2)
		return 0;
	return -1;
#else
#error No balancing possible.
#endif
#endif
}

#ifdef AVL_COUNT
unsigned int avl_count(const avl_tree_t *avltree) {
	return NODE_COUNT(avltree->top);
}

avl_node_t *avl_at(const avl_tree_t *avltree, unsigned int index) {
	avl_node_t *avlnode;
	unsigned int c;

	avlnode = avltree->top;

	while(avlnode) {
		c = L_COUNT(avlnode);

		if(index < c) {
			avlnode = avlnode->left;
		} else if(index > c) {
			avlnode = avlnode->right;
			index -= c+1;
		} else {
			return avlnode;
		}
	}
	return NULL;
}

unsigned int avl_index(const avl_node_t *avlnode) {
	avl_node_t *next;
	unsigned int c;

	c = L_COUNT(avlnode);

	while((next = avlnode->parent)) {
		if(avlnode == next->right)
			c += L_COUNT(next) + 1;
		avlnode = next;
	}

	return c;
}
#endif

int avl_search_closest(const avl_tree_t *avltree, const void *item, avl_node_t **avlnode) {
	avl_node_t *node;
	avl_compare_t cmp;
	int c;

	if(!avlnode)
		avlnode = &node;

	node = avltree->top;

	if(!node)
		return *avlnode = NULL, 0;

	cmp = avltree->cmp;

	for(;;) {
		c = cmp(item, node->item);

		if(c < 0) {
			if(node->left)
				node = node->left;
			else
				return *avlnode = node, -1;
		} else if(c > 0) {
			if(node->right)
				node = node->right;
			else
				return *avlnode = node, 1;
		} else {
			return *avlnode = node, 0;
		}
	}
}

/*
 * avl_search:
 * Return a pointer to a node with the given item in the tree.
 * If no such item is in the tree, then NULL is returned.
 */
avl_node_t *avl_search(const avl_tree_t *avltree, const void *item) {
	avl_node_t *node;
	return avl_search_closest(avltree, item, &node) ? NULL : node;
}

avl_tree_t *avl_init_tree(avl_tree_t *rc, avl_compare_t cmp, avl_freeitem_t freeitem) {
	if(rc) {
		rc->head = NULL;
		rc->tail = NULL;
		rc->top = NULL;
		rc->cmp = cmp;
		rc->freeitem = freeitem;
	}
	return rc;
}

avl_tree_t *avl_alloc_tree(avl_compare_t cmp, avl_freeitem_t freeitem) {
	return (avl_tree_t*) avl_init_tree((avl_tree_t*)ini_malloc(sizeof(avl_tree_t)), cmp, freeitem);
}

void avl_clear_tree(avl_tree_t *avltree) {
	avltree->top = avltree->head = avltree->tail = NULL;
}

void avl_free_nodes(avl_tree_t *avltree) {
	avl_node_t *node, *next;
	avl_freeitem_t freeitem;

	freeitem = avltree->freeitem;

	for(node = avltree->head; node; node = next) {
		next = node->next;
		if(freeitem)
			freeitem(node->item);
		ini_free(node,true);
	}
	avl_clear_tree(avltree);
}

/*
 * avl_free_tree:
 * Free all memory used by this tree.  If freeitem is not NULL, then
 * it is assumed to be a destructor for the items referenced in the avl_
 * tree, and they are deleted as well.
 */
void avl_free_tree(avl_tree_t *avltree) {
	avl_free_nodes(avltree);
	ini_free(avltree, true);
}

static void avl_clear_node(avl_node_t *newnode) {
	newnode->left = newnode->right = NULL;
	#ifdef AVL_COUNT
	newnode->count = 1;
	#endif
	#ifdef AVL_DEPTH
	newnode->depth = 1;
	#endif
}

avl_node_t *avl_init_node(avl_node_t *newnode, void *item) {
	if(newnode) {
	  avl_clear_node(newnode);
	  newnode->item = item;
	}
	return newnode;
}

avl_node_t *avl_insert_top(avl_tree_t *avltree, avl_node_t *newnode) {
	avl_clear_node(newnode);
	newnode->prev = newnode->next = newnode->parent = NULL;
	avltree->head = avltree->tail = avltree->top = newnode;
	return newnode;
}

avl_node_t *avl_insert_before(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode) {
	if(!node)
		return avltree->tail
			? avl_insert_after(avltree, avltree->tail, newnode)
			: avl_insert_top(avltree, newnode);

	if(node->left)
		return avl_insert_after(avltree, node->prev, newnode);

	avl_clear_node(newnode);

	newnode->next = node;
	newnode->parent = node;

	newnode->prev = node->prev;
	if(node->prev)
		node->prev->next = newnode;
	else
		avltree->head = newnode;
	node->prev = newnode;

	node->left = newnode;
	avl_rebalance(avltree, node);
	return newnode;
}

avl_node_t *avl_insert_after(avl_tree_t *avltree, avl_node_t *node, avl_node_t *newnode) {
	if(!node)
		return avltree->head
			? avl_insert_before(avltree, avltree->head, newnode)
			: avl_insert_top(avltree, newnode);

	if(node->right)
		return avl_insert_before(avltree, node->next, newnode);

	avl_clear_node(newnode);

	newnode->prev = node;
	newnode->parent = node;

	newnode->next = node->next;
	if(node->next)
		node->next->prev = newnode;
	else
		avltree->tail = newnode;
	node->next = newnode;

	node->right = newnode;
	avl_rebalance(avltree, node);
	return newnode;
}

avl_node_t *avl_insert_node(avl_tree_t *avltree, avl_node_t *newnode) {
	avl_node_t *node;

	if(!avltree->top)
		return avl_insert_top(avltree, newnode);

	switch(avl_search_closest(avltree, newnode->item, &node)) {
		case -1:
			return avl_insert_before(avltree, node, newnode);
		case 1:
			return avl_insert_after(avltree, node, newnode);
	}

	return NULL;
}

/*
 * avl_insert:
 * Create a new node and insert an item there.
 * Returns the new node on success or NULL if no memory could be allocated.
 */
avl_node_t *avl_insert(avl_tree_t *avltree, void *item) {
	avl_node_t *newnode;

	newnode = avl_init_node((avl_node_t*)ini_malloc(sizeof(avl_node_t)), item);
	if(newnode) {
		if(avl_insert_node(avltree, newnode))
			return newnode;
		ini_free(newnode, true);
		errno = EEXIST;
	}
	return NULL;
}

/*
 * avl_unlink_node:
 * Removes the given node.  Does not delete the item at that node.
 * The item of the node may be freed before calling avl_unlink_node.
 * (In other words, it is not referenced by this function.)
 */
void avl_unlink_node(avl_tree_t *avltree, avl_node_t *avlnode) {
	avl_node_t *parent;
	avl_node_t **superparent;
	avl_node_t *subst, *left, *right;
	avl_node_t *balnode;

	if(avlnode->prev)
		avlnode->prev->next = avlnode->next;
	else
		avltree->head = avlnode->next;

	if(avlnode->next)
		avlnode->next->prev = avlnode->prev;
	else
		avltree->tail = avlnode->prev;

	parent = avlnode->parent;

	superparent = parent
		? avlnode == parent->left ? &parent->left : &parent->right
		: &avltree->top;

	left = avlnode->left;
	right = avlnode->right;
	if(!left) {
		*superparent = right;
		if(right)
			right->parent = parent;
		balnode = parent;
	} else if(!right) {
		*superparent = left;
		left->parent = parent;
		balnode = parent;
	} else {
		subst = avlnode->prev;
		if(subst == left) {
			balnode = subst;
		} else {
			balnode = subst->parent;
			balnode->right = subst->left;
			if(balnode->right)
				balnode->right->parent = balnode;
			subst->left = left;
			left->parent = subst;
		}
		subst->right = right;
		subst->parent = parent;
		right->parent = subst;
		*superparent = subst;
	}

	avl_rebalance(avltree, balnode);
}

void *avl_delete_node(avl_tree_t *avltree, avl_node_t *avlnode) {
	void *item = NULL;
	if(avlnode) {
		item = avlnode->item;
		avl_unlink_node(avltree, avlnode);
		if(avltree->freeitem)
			avltree->freeitem(item);
		ini_free(avlnode,true);
	}
	return item;
}

void *avl_delete(avl_tree_t *avltree, const void *item) {
	return avl_delete_node(avltree, avl_search(avltree, item));
}

avl_node_t *avl_fixup_node(avl_tree_t *avltree, avl_node_t *newnode) {
	avl_node_t *oldnode = NULL, *node;

	if(!avltree || !newnode)
		return NULL;

	node = newnode->prev;
	if(node) {
		oldnode = node->next;
		node->next = newnode;
	} else {
		avltree->head = newnode;
	}

	node = newnode->next;
	if(node) {
		oldnode = node->prev;
		node->prev = newnode;
	} else {
		avltree->tail = newnode;
	}

	node = newnode->parent;
	if(node) {
		if(node->left == oldnode)
			node->left = newnode;
		else
			node->right = newnode;
	} else {
		oldnode = avltree->top;
		avltree->top = newnode;
	}

	return oldnode;
}

/*
 * avl_rebalance:
 * Rebalances the tree if one side becomes too heavy.  This function
 * assumes that both subtrees are AVL-trees with consistant data.  The
 * function has the additional side effect of recalculating the count of
 * the tree at this node.  It should be noted that at the return of this
 * function, if a rebalance takes place, the top of this subtree is no
 * longer going to be the same node.
 */
void avl_rebalance(avl_tree_t *avltree, avl_node_t *avlnode) {
	avl_node_t *child;
	avl_node_t *gchild;
	avl_node_t *parent;
	avl_node_t **superparent;

	parent = avlnode;

	while(avlnode) {
		parent = avlnode->parent;

		superparent = parent
			? avlnode == parent->left ? &parent->left : &parent->right
			: &avltree->top;

		switch(avl_check_balance(avlnode)) {
		case -1:
			child = avlnode->left;
			#ifdef AVL_DEPTH
			if(L_DEPTH(child) >= R_DEPTH(child)) {
			#else
			#ifdef AVL_COUNT
			if(L_COUNT(child) >= R_COUNT(child)) {
			#else
			#error No balancing possible.
			#endif
			#endif
				avlnode->left = child->right;
				if(avlnode->left)
					avlnode->left->parent = avlnode;
				child->right = avlnode;
				avlnode->parent = child;
				*superparent = child;
				child->parent = parent;
				#ifdef AVL_COUNT
				avlnode->count = CALC_COUNT(avlnode);
				child->count = CALC_COUNT(child);
				#endif
				#ifdef AVL_DEPTH
				avlnode->depth = CALC_DEPTH(avlnode);
				child->depth = CALC_DEPTH(child);
				#endif
			} else {
				gchild = child->right;
				avlnode->left = gchild->right;
				if(avlnode->left)
					avlnode->left->parent = avlnode;
				child->right = gchild->left;
				if(child->right)
					child->right->parent = child;
				gchild->right = avlnode;
				if(gchild->right)
					gchild->right->parent = gchild;
				gchild->left = child;
				if(gchild->left)
					gchild->left->parent = gchild;
				*superparent = gchild;
				gchild->parent = parent;
				#ifdef AVL_COUNT
				avlnode->count = CALC_COUNT(avlnode);
				child->count = CALC_COUNT(child);
				gchild->count = CALC_COUNT(gchild);
				#endif
				#ifdef AVL_DEPTH
				avlnode->depth = CALC_DEPTH(avlnode);
				child->depth = CALC_DEPTH(child);
				gchild->depth = CALC_DEPTH(gchild);
				#endif
			}
		break;
		case 1:
			child = avlnode->right;
			#ifdef AVL_DEPTH
			if(R_DEPTH(child) >= L_DEPTH(child)) {
			#else
			#ifdef AVL_COUNT
			if(R_COUNT(child) >= L_COUNT(child)) {
			#else
			#error No balancing possible.
			#endif
			#endif
				avlnode->right = child->left;
				if(avlnode->right)
					avlnode->right->parent = avlnode;
				child->left = avlnode;
				avlnode->parent = child;
				*superparent = child;
				child->parent = parent;
				#ifdef AVL_COUNT
				avlnode->count = CALC_COUNT(avlnode);
				child->count = CALC_COUNT(child);
				#endif
				#ifdef AVL_DEPTH
				avlnode->depth = CALC_DEPTH(avlnode);
				child->depth = CALC_DEPTH(child);
				#endif
			} else {
				gchild = child->left;
				avlnode->right = gchild->left;
				if(avlnode->right)
					avlnode->right->parent = avlnode;
				child->left = gchild->right;
				if(child->left)
					child->left->parent = child;
				gchild->left = avlnode;
				if(gchild->left)
					gchild->left->parent = gchild;
				gchild->right = child;
				if(gchild->right)
					gchild->right->parent = gchild;
				*superparent = gchild;
				gchild->parent = parent;
				#ifdef AVL_COUNT
				avlnode->count = CALC_COUNT(avlnode);
				child->count = CALC_COUNT(child);
				gchild->count = CALC_COUNT(gchild);
				#endif
				#ifdef AVL_DEPTH
				avlnode->depth = CALC_DEPTH(avlnode);
				child->depth = CALC_DEPTH(child);
				gchild->depth = CALC_DEPTH(gchild);
				#endif
			}
		break;
		default:
			#ifdef AVL_COUNT
			avlnode->count = CALC_COUNT(avlnode);
			#endif
			#ifdef AVL_DEPTH
			avlnode->depth = CALC_DEPTH(avlnode);
			#endif
		}
		avlnode = parent;
	}
}

typedef struct dlnode {
  double *x;                    /* The data vector              */
  struct dlnode **next;         /* Next-node vector             */
  struct dlnode **prev;         /* Previous-node vector         */
  struct avl_node_t * tnode;
  int ignore;
#if VARIANT >= 2
  double *area;                 /* Area */
#endif
#if VARIANT >= 3
  double *vol;                  /* Volume */
#endif
} dlnode_t;


static avl_tree_t *tree;
//extern int stop_dimension;


static int compare_node( const void *p1, const void* p2)
{
    const double x1 = *((*(const dlnode_t **)p1)->x);
    const double x2 = *((*(const dlnode_t **)p2)->x);

    if ( x1 == x2 )
        return 0;
    else
        return ( x1 < x2 ) ? -1 : 1;
}

static int compare_tree_asc( const void *p1, const void *p2)
{
    const double x1= *((const double *)p1+1);
    const double x2= *((const double *)p2+1);

    if (x1 != x2)
        return (x1 > x2) ? -1 : 1;
    else
        return 0;
}


/*
 * Setup circular double-linked list in each dimension
 */

static dlnode_t *
setup_cdllist(double *data, int d, int n)
{
    dlnode_t *head;
    dlnode_t **scratch;
    int i, j;

    head  = (dlnode_t*) ini_malloc ((n+1) * sizeof(dlnode_t));

    head->x = data;
    head->ignore = 0;  /* should never get used */
    head->next = (dlnode**) ini_malloc( d * (n+1) * sizeof(dlnode_t*));
    head->prev = (dlnode**) ini_malloc( d * (n+1) * sizeof(dlnode_t*));
    head->tnode = (avl_node_t*) ini_malloc( sizeof(avl_node_t));

#if VARIANT >= 2
    head->area = (double*) ini_malloc(d * (n+1) * sizeof(double));
#endif
#if VARIANT >= 3
    head->vol = (double*) ini_malloc(d * (n+1) * sizeof(double));
#endif

    for (i = 1; i <= n; i++) {
        head[i].x = head[i-1].x + d ;/* this will be fixed a few lines below... */
        head[i].ignore = 0;
        head[i].next = head[i-1].next + d;
        head[i].prev = head[i-1].prev + d;
        head[i].tnode = (avl_node_t*) ini_malloc(sizeof(avl_node_t));
#if VARIANT >= 2
        head[i].area = head[i-1].area + d;
#endif
#if VARIANT >= 3
        head[i].vol = head[i-1].vol + d;
#endif
    }
    head->x = NULL; /* head contains no data */

    scratch = (dlnode_t**)ini_malloc(n * sizeof(dlnode_t*));
    for (i = 0; i < n; i++)
        scratch[i] = head + i + 1;

    for (j = d-1; j >= 0; j--) {
        for (i = 0; i < n; i++)
            scratch[i]->x--;
        qsort(scratch, n, sizeof(dlnode_t*), compare_node);
        head->next[j] = scratch[0];
        scratch[0]->prev[j] = head;
        for (i = 1; i < n; i++) {
            scratch[i-1]->next[j] = scratch[i];
            scratch[i]->prev[j] = scratch[i-1];
        }
        scratch[n-1]->next[j] = head;
        head->prev[j] = scratch[n-1];
    }

    /* free(scratch); */

    return head;
}

static void deleteNode (dlnode_t *nodep, int dim, double * bound )
{
    int i;

    for (i = 0; i < dim; i++) {
        nodep->prev[i]->next[i] = nodep->next[i];
        nodep->next[i]->prev[i] = nodep->prev[i];
#if VARIANT >= 3
        if (bound[i] > nodep->x[i])
            bound[i] = nodep->x[i];
#endif
  }
}

static void reinsert (dlnode_t *nodep, int dim, double * bound )
{
    int i;

    for (i = 0; i < dim; i++) {
        nodep->prev[i]->next[i] = nodep;
        nodep->next[i]->prev[i] = nodep;
#if VARIANT >= 3
        if (bound[i] > nodep->x[i])
            bound[i] = nodep->x[i];
#endif
    }
}

static double
hv_recursive( dlnode_t *list, int dim, int c, const double * ref,
              double * bound)
{
    dlnode_t *p0,*p1,*pp;
    double hypera,hyperv=0;
    double height;


    /* ------------------------------------------------------
       General case for dimensions higher than stop_dimension
       ------------------------------------------------------ */
    if ( dim > stop_dimension ) {
        p0 = list;
#if VARIANT >= 2
        for (p1 = p0->prev[dim]; p1->x; p1 = p1->prev[dim]) {
            if (p1->ignore < dim)
                p1->ignore = 0;
        }
#endif
        while (
#if VARIANT >= 3
            p0->prev[dim]->x[dim] > bound[dim] &&
#endif
            c > 1 ) {
            p0 = p0->prev[dim];
            deleteNode(p0, dim, bound);
            c--;
        }
        p1 = p0->prev[dim];

#if VARIANT == 1
        hypera = hv_recursive(list, dim-1, c, ref, bound);
#elif VARIANT >= 3
        if (c > 1)
            hyperv = p1->prev[dim]->vol[dim] + p1->prev[dim]->area[dim]
                * (p1->x[dim] - p1->prev[dim]->x[dim]);
        else {
            p1->area[0] = 1;
            int i;
            for (i = 1; i <= dim; i++)
                p1->area[i] = p1->area[i-1] * (ref[i-1] - p1->x[i-1]);
        }
        p1->vol[dim] = hyperv;
#endif
#if VARIANT >= 2
        if (p1->ignore >= dim)
            p1->area[dim] = p1->prev[dim]->area[dim];
        else {
            p1->area[dim] = hv_recursive(list, dim-1, c, ref, bound);
            if (p1->area[dim] <= p1->prev[dim]->area[dim])
                p1->ignore = dim;
        }
#endif

        while (p0->x != NULL) {
            hyperv +=
#if VARIANT == 1
                hypera
#elif VARIANT >= 2
                p1->area[dim]
#endif
                * (p0->x[dim] - p1->x[dim]);
#if VARIANT >= 3
            bound[dim] = p0->x[dim];
#endif
            reinsert(p0, dim, bound);
            c++;
            p1 = p0;
            p0 = p0->next[dim];
#if VARIANT >= 3
            p1->vol[dim] = hyperv;
#endif
#if VARIANT == 1
            hypera = hv_recursive(list, dim-1, c, ref, NULL);
#elif VARIANT >= 2
            if (p1->ignore >= dim)
                p1->area[dim] = p1->prev[dim]->area[dim];
            else {
                p1->area[dim] = hv_recursive(list, dim-1, c, ref, bound);
                if (p1->area[dim] <= p1->prev[dim]->area[dim])
                    p1->ignore = dim;
            }
#endif
        }
        hyperv +=
#if VARIANT == 1
            hypera
#elif VARIANT >= 2
            p1->area[dim]
#endif
            * (ref[dim] - p1->x[dim]);

        return hyperv;
    }

    /* ---------------------------
       special case of dimension 3
       --------------------------- */
    else if (dim == 2) {

        pp = list->next[2];
        avl_init_node(pp->tnode,pp->x);
        avl_insert_top(tree,pp->tnode);

        hypera = (ref[0] - pp->x[0])*(ref[1] - pp->x[1]);

        if (c == 1)
            height = ref[2] - pp->x[2];
        else
            height = pp->next[2]->x[2] - pp->x[2];

        hyperv = hypera * height;

        while ((pp=pp->next[2])->x) {
            height = (pp==list->prev[2])
                ? ref[2] - pp->x[2]
                : pp->next[2]->x[2] - pp->x[2];
#if VARIANT >= 2
            if (pp->ignore>=2)
                hyperv += hypera * height;
            else {
#endif
                const double * prv_ip, * nxt_ip;
                avl_node_t *tnode;

                avl_init_node(pp->tnode, pp->x);

                if (avl_search_closest(tree, pp->x, &tnode) <= 0) {
                    nxt_ip = (double *)(tnode->item);
                    tnode = tnode->prev;
                } else {
                    nxt_ip = (tnode->next!=NULL)
                        ? (double *)(tnode->next->item)
                        : ref;
                }

                if (nxt_ip[0] > pp->x[0]) {

                    avl_insert_after(tree, tnode, pp->tnode);

                    if (tnode !=NULL) {
                        prv_ip = (double *)(tnode->item);

                        if (prv_ip[0] > pp->x[0]) {
                            const double * cur_ip;

                            tnode = pp->tnode->prev;
                            // cur_ip = point dominated by pp with highest [0]-coordinate
                            cur_ip = (double *)(tnode->item);
                            while (tnode->prev) {
                                prv_ip = (double *)(tnode->prev->item);
                                hypera -= (prv_ip[1] - cur_ip[1])*(nxt_ip[0] - cur_ip[0]);
                                if (prv_ip[0] < pp->x[0])
                                    break; // prv is not dominated by pp
                                cur_ip = prv_ip;
                                avl_unlink_node(tree,tnode);
                                tnode = tnode->prev;
                            }

                            avl_unlink_node(tree,tnode);

                            if (!tnode->prev) {
                                hypera -= (ref[1] - cur_ip[1])*(nxt_ip[0] - cur_ip[0]);
                                prv_ip = ref;
                            }
                        }
                    } else
                        prv_ip = ref;

                    hypera += (prv_ip[1] - pp->x[1])*(nxt_ip[0] - pp->x[0]);

                }
                else
                    pp->ignore = 2;

                if (height > 0)
                    hyperv += hypera * height;

#if VARIANT >= 2
            }
#endif
        }
        avl_clear_tree(tree);
        return hyperv;
    }

    /* special case of dimension 2 */
    else if (dim == 1) {
        p1 = list->next[1];
        hypera = p1->x[0];
        while ((p0 = p1->next[1])->x) {
            hyperv += (ref[0] - hypera) * (p0->x[1] - p1->x[1]);
            if (p0->x[0] < hypera)
                hypera = p0->x[0];
            p1 = p0;
        }
        hyperv += (ref[0] - hypera) * (ref[1] - p1->x[1]);
        return hyperv;
    }

    /* special case of dimension 1 */
    else if (dim == 0) {
        return (ref[0] - list->next[0]->x[0]);
    }

    else {
        fprintf(stderr, "%s:%d: unreachable condition! \n"
                "This is a bug, please report it to "
                "m.lopez-ibanez@napier.ac.uk\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

double fonseca(double *data, double *ref, unsigned int d, unsigned int n)
{
    dlnode_t *list;
    double hyperv;
    double * bound = NULL;

#if VARIANT >= 3
    unsigned int i;

    bound = (double*)ini_malloc (d * sizeof(double));
    for (i = 0; i < d; i++) bound[i] = -std::numeric_limits<double>::max();
#endif

    tree  = avl_alloc_tree ((avl_compare_t) compare_tree_asc,
                            (avl_freeitem_t) free);

    list = setup_cdllist(data, d, n);

    hyperv = hv_recursive(list, d-1, n, ref, bound);


	/* Manually free all allocated memory - kind of garbage collection */
	for (i = 0; i < allocated_pointers.size(); i++ ) {
		free( allocated_pointers[i] );
	}

	allocated_pointers.clear();
    return hyperv;
}


struct LastObjectiveComparator {
        static unsigned int NO_OBJECTIVES;
        static int compare( const void * p1, const void * p2 ) {
                const double * d1 = reinterpret_cast<const double*>( p1 );
                const double * d2 = reinterpret_cast<const double*>( p2 );

                if (d1[NO_OBJECTIVES-1] < d2[NO_OBJECTIVES-1]) return -1;
                        else if (d1[NO_OBJECTIVES-1] > d2[NO_OBJECTIVES-1]) return 1;
                        else return 0;
                }
        };
unsigned int LastObjectiveComparator::NO_OBJECTIVES = 0;

double hypervolume(double* points, double* referencePoint, unsigned int noObjectives, unsigned int noPoints) {
        unsigned int i;
        if (noObjectives == 0) {
                throw std::invalid_argument("[hypervolume] dimension must be positive");
        }
        else if (noObjectives == 1) {
                // trivial
                double m = 1e100;
                for (i=0; i<noPoints; i++) if (points[i] < m) m = points[i];
                double h = *referencePoint - m;
                if (h < 0.0) h = 0.0;
                return h;
        }
        else if (noObjectives == 2) {
                // sort by last objective
                LastObjectiveComparator::NO_OBJECTIVES = 2;
                qsort(points, noPoints, noObjectives * sizeof(double), LastObjectiveComparator::compare );
                // sortByLastObjective(points, noObjectives, noPoints);
                double h = (referencePoint[0] - points[0]) * (referencePoint[1] - points[1]);
                double diffDim1;
                unsigned int pre = 0;
                for (i=1; i<noPoints; i++) 
                {
                        diffDim1 = points[2*pre] - points[2*i];  // Might be negative, if the i-th solution is dominated.
                        if( diffDim1 > 0 )
                        {
							h += diffDim1 * (referencePoint[1] - points[2*i+1]);
							pre = i;
						}
                }
                return h;
        }
        else if (noObjectives == 3) {
                return fonseca(points, referencePoint, noObjectives, noPoints);
        }
        else {
                LastObjectiveComparator::NO_OBJECTIVES = noObjectives;
                qsort(points, noPoints, noObjectives * sizeof(double), LastObjectiveComparator::compare );
                // sortByLastObjective(points, noObjectives, noPoints);
                return overmars_yap(points, referencePoint, noObjectives, noPoints);
        }
}

