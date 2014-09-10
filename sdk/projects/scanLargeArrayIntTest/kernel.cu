#include "paracuda.h"
#include "paracuda_segmentedScan.h"
#include "limits.h"

/* Define the mapping (x, y) -> (x + y, x * y) */

PARACUDA_STRUCT_2(pair_t, pair_vector_t,
    int, x, 
    int, y
)

PARACUDA_OPERATOR(plus_times, void, (pair_t* r, pair_t* v), ({
    r->x = v->x + v->y;
    r->y = v->x * v->y;
}))

PARACUDA_MAP(plus_times_map, plus_times, pair_t, pair_t)

/* Define the scan that can detect ordering */

PARACUDA_OPERATOR(ordered_zero, void, (pair_t* r), ({
    r->x = 0;
    r->y = 2;
}))

PARACUDA_OPERATOR(ordered, void, (pair_t* r, pair_t* a, pair_t* b), ({
    if(a->y == 2) {
        r->x = b->x;
        r->y = 1;
        return;
    }
    r->x = b->x;
    r->y = a->x <= b->x && a->y && b->y;
}))

PARACUDA_SCAN(ordered_scan, ordered, ordered_zero, pair_t)

/* Define the scan [a, b, ..., y, z] -> [0, a, a + b, ..., a + b + ... + y] */

PARACUDA_SINGLE(int)

PARACUDA_OPERATOR(plus, void, (int* r, int* a, int* b), ({
    *r = *a + *b;
}))

PARACUDA_OPERATOR(zero, void, (int* r), ({
    *r = 0;
}))

PARACUDA_OPERATOR(one, void, (int* r), ({
   *r = 1;
}));

PARACUDA_SCAN(plus_scan, plus, zero, int)

/* Define segmented scan via scan (experiment, non-working - ys are being set to non-zero by an evil magical force) */

PARACUDA_OPERATOR(int_flag_zero, void, (pair_t* r), ({
    r->x = 0;
}))

PARACUDA_OPERATOR(int_flag_zero_seg, void, (int* r), ({
  *r = 0;
}))

PARACUDA_OPERATOR(segmented_plus, void, (int* r, int* a, int* b), ({
    *r = *a + *b; 
}))

PARACUDA_OPERATOR(add, void, (int* r, pair_t* i), ({
    *r = i->x + i->y; 
}))

PARACUDA_OPERATOR(minus, void, (int* r, pair_t* i), ({
   *r = i->x - i->y;
}))

PARACUDA_SEGMENTEDSCAN(segmented_plus_scan, segmented_plus, zero, int)

/* Negation [0, 1, 1, 0, 0, 0, 1, 0] -> [1, 0, 0, 1, 1, 1, 0, 1] */

PARACUDA_OPERATOR(negate, void, (int* r, int* a), ({
    *r = (*a) ? 0 : 1;
}))

PARACUDA_MAP(negate_map, negate, int, int)

PARACUDA_STRUCT_3(split_t, split_vector_t,
    int, flags, 
    int, left,
    int, right
)

PARACUDA_OPERATOR(split, void, (int* r, split_t* a), ({
    *r = (a->flags) ? a->right : a->left;	
}))

PARACUDA_MAP(split_map, split, int, split_t)

PARACUDA_PERMUTE(int_permute, int)

// Radix

PARACUDA_STRUCT_2(bitwise_t, bitwise_vector_t,
    int, integer,
    int, number
)

PARACUDA_OPERATOR(int_bitwise, void, (int* o, bitwise_t* r), ({
*o = (r->integer & r->number) == r->number;
}))

PARACUDA_MAP(int_bitwise_map, int_bitwise, int, bitwise_t)


// Transfer

PARACUDA_OPERATOR(int_transfer, void, (int* o, int* i), ({
*o = *i;
}))

PARACUDA_MAP(int_transfer_map, int_transfer, int, int)


PARACUDA_OPERATOR(int_copy_pivot, void, (int* r, pair_t* i), ({
     *r = (i->x) ? i->y : 0;
}))

PARACUDA_MAP(int_copy_pivot_map, int_copy_pivot, int, pair_t)

PARACUDA_OPERATOR(int_add_pivot, void, (int* r, split_t* i), ({
     *r = (i->flags) ? i->left + i->right : i->left;
}))

PARACUDA_MAP(int_add_pivot_map, int_add_pivot, int, split_t)

PARACUDA_OPERATOR(int_insert_pivot, void, (int* r, split_t* i), ({
     *r = (i->flags) ? i->left : i->right;
}))

PARACUDA_MAP(int_insert_pivot_map, int_insert_pivot, int, split_t);

PARACUDA_OPERATOR(and_distribute_op, void, (int* r, int* a, int* b), ({
    *r = (*a & *b);
}))

PARACUDA_OPERATOR(or_distribute_op, void, (int* r, int* a, int* b), ({
    *r = (*a | *b);
}))

PARACUDA_OPERATOR(max, void, (int* r, int* a, int* b), ({
  *r = (*a < *b) ? *b : *a;
}))


PARACUDA_SCAN(and_distribute_scan, and_distribute_op, one, int)

PARACUDA_SCAN(or_distribute_scan, or_distribute_op, zero, int)

PARACUDA_SEGMENTEDSCAN(and_distribute_segmented_scan, and_distribute_op, one, int)

PARACUDA_SEGMENTEDSCAN(or_distribute_segmented_scan, or_distribute_op, one, int)

PARACUDA_SEGMENTEDSCAN(max_segmented_scan, max, zero, int)

PARACUDA_OPERATOR(compare, void, (int* r, pair_t* i), ({
    *r = i->y <= i->x;
}))

PARACUDA_MAP(compare_map, compare, int, pair_t)

PARACUDA_MAP(map_add, add, int, pair_t)

PARACUDA_OPERATOR(map_or, void, (int* r, pair_t* i), ({
   *r = i->x | i->y;  
}))

PARACUDA_MAP(or_map, map_or, int, pair_t)

PARACUDA_MAP(map_minus, minus, int, pair_t)


PARACUDA_OPERATOR(find_new_array, void, (int* r, split_t* i), ({
    *r = ((i->flags < i->left && !(i->flags < i->right))) || ((i->flags > i->left && !(i->flags > i->right))) ? 1 : 0; 
}))

PARACUDA_MAP(find_new_array_map, find_new_array, int, split_t)

