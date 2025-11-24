
#include "datax.h"

#include <stack>
#include <assert.h>

#include "main.h"		// for dbgprintf

DataX::DataX () 
{
	mHeap = 0x0;
	mHeapNum = 0;
	mHeapMax = 0;
	mHeapFree = 0;
	for (int n=0; n < REF_MAX; n++ ) mRef[n]=BUNDEF;
}
	

DataX::~DataX() 
{
	DeleteAllBuffers ();
}

//--------------- Buffers

void DataX::DeleteAllBuffers ()
{
	for (int n=0; n < (int) mBuf.size(); n++) {
		mBuf[n].Clear ();
	}
	mBuf.clear ();
}

void DataX::ClearHeap ()
{
	if ( mHeap != 0x0 ) free ( mHeap );
	mHeap = 0x0;
	mHeapNum = 0;
	mHeapMax = 0;	
}

// AddBuffer 
int DataX::AddBuffer ( int userid, std::string name, ushort stride, uint64_t maxcnt, uchar dest_flags )
{
	DataPtr buf;
	buf.mRefID = userid;
	buf.SetUsage ( DT_MISC, dest_flags, Vector3DI(maxcnt,1,1) );
	buf.Resize ( stride, maxcnt, 0x0, dest_flags );

	// add to buffer list
	int b = (int) mBuf.size();		
	mBuf.push_back ( buf );
	mRef[ userid ] = b;
	return b;
}
void DataX::SetBufferUsage	( int i, uchar dt, uchar use_flags, Vector3DI res  )
{
	int b = mRef[i];  if (b==BUNDEF) return;
	mBuf[b].SetUsage ( dt, use_flags, res );
}

void DataX::SetNum ( int num )
{
	for (int b=0; b < mBuf.size(); b++) 
		mBuf[b].mNum = num;		
}

void DataX::AssignToGPU ( std::string var_name, CUmodule& module )
{
	size_t len;
	char varname[1024];
	strcpy ( varname, var_name.c_str() );

	cuCheck ( cuModuleGetGlobal ( &cuData, &len,	module, (const char*) varname ), "LoadKernel", "cuModuleGetGlobal", varname, true );
	
	if ( len != sizeof(cuDataX) ) {
		dbgprintf ( "ERROR: AssignToGPU. Size of GPU symbol does not match size of cuDataX.\n" );
		exit(-12);
	}
}
void DataX::UpdateGPUAccess ()
{
	int userid;
	cuDataX datax;		
	memset ( &datax, 0, sizeof(cuDataX) );

	//dbgprintf ("------\n" );
	for (int i=0; i < mBuf.size(); i++) {		
		userid = mBuf[i].mRefID;
		datax.mbuf[ userid ] = mBuf[i].mGpu;
		//dbgprintf ("  %d, cpu: %012llx, gpu: %012llx\n", i, cpu(b), gpu(b) );			// debugging		
	}
	cuCheck ( cuMemcpyHtoD( cuData,	&datax, sizeof(cuDataX) ), "UpdateGPUAccess", "cuMemcpyHtoD", "cuData", false );
}
void DataX::MatchAllBuffers ( DataX* src, uchar use_flags )
{
	int userid;

	DeleteAllBuffers ();
	
	for (int i=0; i < src->mBuf.size(); i++) {		
		userid = src->mBuf[i].mRefID;
		AddBuffer ( userid, "", src->mBuf[i].mStride, src->mBuf[i].mMax, (use_flags!=DT_MISC) ? use_flags : src->mBuf[i].mUseFlags );
		SetBufferUsage ( userid, src->mBuf[i].mUseType );
	}
}

void DataX::CopyAllBuffers ( DataX* dest, uchar dest_flags )
{
	// copy all buffers between two DataX
	int b; 

	for (int b=0; b < mBuf.size(); b++ )  {				
		if ( mBuf[b].mRefID != dest->mBuf[b].mRefID) {
			dbgprintf ( "ERROR: CopyAllBuffers. RefIDs do not match.\n" );
			exit(-9);
		}
		CopyBuffer ( b, b, dest, dest_flags );
	}
}



void DataX::Commit ( int i )
{
	int b = mRef[i];  if (b==BUNDEF) return;
	mBuf[b].Commit ();
}
void DataX::Retrieve ( int i )
{
	int b = mRef[i];  if (b==BUNDEF) return;
	mBuf[b].Retrieve();
}
void DataX::CommitAll ()
{
	for (int b=0; b < mBuf.size(); b++)
		mBuf[b].Commit ();
}
char* DataX::printElem ( int i, int n, char* buf )
{
	int b = mRef[i]; 
	if ( b==BUNDEF) {
		sprintf ( buf, "buf not found" ); 
		return buf;
	}

	switch (mBuf[b].mUseType) {
	case DT_MISC:	sprintf ( buf, "unknown use type" ); break;		
	case DT_UINT:	sprintf ( buf, "%u", *((uint*) mBuf[b].mCpu + n) );		break;
	case DT_INT:	sprintf ( buf, "%d",  *((int*) mBuf[b].mCpu + n) );		break;
	case DT_FLOAT:	sprintf ( buf, "%f",  *((float*) mBuf[b].mCpu + n) );	break;
	case DT_FLOAT3: {
		Vector3DF v = *((Vector3DF*) mBuf[b].mCpu + n);
		sprintf ( buf, "%f,%f,%f", v.x, v.y, v.z );	} break;
	case DT_FLOAT4: {
		Vector4DF v = *((Vector4DF*) mBuf[b].mCpu + n);
		sprintf ( buf, "%f,%f,%f,%f", v.x, v.y, v.z, v.w );	} break;	
	};
	return buf;
}


// ExpandBuffer
//     | 1 | 2 | 3 | .. | N |  |  | M |<---- used ---->|  unused   |maxsize
//  num		- number of elements of fixed stride
//  maxcnt	- maximum number of elements (M)
//  used	- amount of additional used data in bytes
//  maxsize	- maximum size of buffer in bytes (currently allocated)
//
char* DataX::ExpandBuffer ( int i, int max_cnt )
{
	int b=mRef[i]; if ( b==BUNDEF) return 0;

	// expand buffer (mexpand does malloc, copy and free)
	int new_size = max_cnt*mBuf[b].mStride;
	char* new_data = (char*) malloc ( new_size );	
	if ( mBuf[b].mCpu != 0x0 ) {
		memcpy ( new_data, mBuf[b].mCpu, mBuf[b].mNum*mBuf[b].mStride );
		free ( mBuf[b].mCpu );
	}	
	mBuf[b].mMax = max_cnt;
	mBuf[b].mSize = new_size;	
	mBuf[b].mCpu = new_data;

	return new_data;
}

// Add Element
int DataX::AddElem ( int i )
{
	int b=mRef[i]; if ( b==BUNDEF ) return -1;

	if ( mBuf[b].mNum >= mBuf[b].mMax ) {
		mBuf[b].mMax = (mBuf[b].mMax==0) ? 1 : mBuf[b].mMax * 2;
		ExpandBuffer ( i, mBuf[b].mMax );			
	}
	mBuf[b].mNum++;
	return mBuf[b].mNum-1;
}

void DataX::EmptyBuffers (int max_cnt)
{
	for (int n=0; n < mBuf.size(); n++) 
		EmptyBuffer ( n, max_cnt);
}

void DataX::EmptyBuffer ( int i, int max_cnt )
{	
	int b=mRef[i]; if (b==BUNDEF ) return;

	// Clear element buffer			
	if ( mBuf[b].mMax < max_cnt) {
		mBuf[b].mMax = max_cnt;		
		mBuf[b].mSize = mBuf[b].mMax*mBuf[b].mStride;
		if (mBuf[b].mCpu != 0x0 ) free ( mBuf[b].mCpu );
		mBuf[b].mCpu = (char*) malloc ( mBuf[b].mSize );
	}
	mBuf[b].mNum = 0;
}

void DataX::CopyBuffer ( uchar bsrc, uchar bdest, DataX* dest, uchar dest_flags )
{
	// copy buffer from one DataX into buffer in another DataX
	mBuf[bsrc].CopyTo ( &dest->mBuf[bdest], dest_flags );
}

int DataX::GetHeapSize ()
{
	int sum = mHeapNum * sizeof(hval) ;
	//for (int n=0; n < (int) mBuf.size(); n++)
//		sum += mBuf[n].size;	
	//sum += (int) mAttribute.size() * sizeof(GeomAttr);
	return sum;
}

// Add specific number of elements
char* DataX::AddElem ( int i, int cnt )
{
	int b=mRef[i];	if (b==BUNDEF) return 0;

	int n = mBuf[b].mNum;
	if ( n + cnt >= mBuf[b].mMax ) {	
		mBuf[b].mMax += cnt;
		int new_size = mBuf[b].mMax * mBuf[b].mStride;
		char* new_data = (char*) malloc ( new_size );		
		if ( mBuf[b].mCpu != 0x0 ) free ( mBuf[b].mCpu );
		mBuf[b].mCpu = new_data;		
	}
	mBuf[b].mNum += cnt;
	return mBuf[b].mCpu + (n * mBuf[b].mStride);
}

// Get a random element in buffer
char* DataX::RandomElem ( uchar i, href& ndx )
{
	int b=mRef[i]; if ( b==BUNDEF) return 0;
	ndx = ( (double) mBuf[i].mNum * (double) rand()) / (double) RAND_MAX;
	return mBuf[b].mCpu + ndx*mBuf[b].mStride;
}

// Delete element in buffer
bool DataX::DelElem ( uchar i, int ndx )
{
	int b=mRef[i]; if ( b==BUNDEF) return 0;
	if ( ndx < 0 || ndx >= mBuf[b].mNum ) return false;
	memcpy ( mBuf[b].mCpu + ndx*mBuf[b].mStride, mBuf[b].mCpu + (ndx+1)*mBuf[b].mStride, (mBuf[b].mNum-1-ndx)*mBuf[b].mStride );		
	mBuf[b].mNum--;	
	return true;
}

//---------------------------------------------------------------- HEAP
void DataX::ResetHeap ()
{
	mHeapNum = 0;
	mHeapFree = -1;
}

void DataX::AddHeap ( int max )
{
	mHeap = (hval*) malloc ( max * sizeof(hval ) );
	mHeapMax = max;
	mHeapNum = 0;
	mHeapFree = -1;
}

hval* DataX::GetHeap ( hpos& num, hpos& max, hpos& free )
{
	num = mHeapNum;
	max = mHeapMax;
	free = mHeapFree;
	return mHeap;
}

void DataX::CopyHeap ( DataX& src )
{
	hpos num, max, freepos;
	hval* src_data = src.GetHeap ( num, max, freepos );

	if ( mHeap != 0x0 ) {
		free ( mHeap );
		mHeap = 0x0;
		mHeapNum = 0;
		mHeapMax = 0;
	}

	if ( max > 0 ) {
		mHeap = (hval*) malloc ( max * sizeof(hval));
		mHeapMax = max;
		mHeapNum = num;
		mHeapFree = freepos;
		memcpy ( mHeap, src_data, mHeapNum * sizeof(hval) );
	}
}

void DataX::ClearRefs ( hList* list )
{
	list->cnt = 0;
	list->max = 0;
	list->pos = 0;
}

hval DataX::AddRef ( hval r, hList* list, hval delta )
{	
	if ( list->max == 0 ) {
		list->cnt = 1;		
		list->pos = HeapAlloc ( HEAP_INIT, list->max );
		*(mHeap + list->pos) = r+delta;
	} else {
		if ( list->cnt >= list->max ) {			
			int siz = list->max;
			hpos new_pos = HeapAlloc ( siz+HEAP_INIT, list->max );				// Alloc new location
			//printf ( "MOVE %d -> %d\n", list.pos, new_pos );			
			memcpy ( mHeap+new_pos, mHeap + list->pos, list->cnt*sizeof(hval) );	// Copy data to new location
			HeapAddFree ( list->pos, siz );										// Free old location						
			list->pos = new_pos;				
		}
		*(mHeap + list->pos + list->cnt) = r+delta;
		list->cnt++;
	}
	return list->pos;
}

void DataX::HeapAddFree ( hpos pos, int size )
{
	memset ( mHeap+pos, 0xFFFF, size*sizeof(hval) );

	if ( pos < mHeapFree || mHeapFree == -1 ) {
		// Lowest position. Insert at head of heap.
		* (hpos*) (mHeap+pos) = size;	
		* (hpos*) (mHeap+pos + HEAP_POS) = mHeapFree;
		mHeapFree = pos;		
		if ( mHeapFree != -1 && *(hpos*) (mHeap+mHeapFree + HEAP_POS) == 0x0 ) { printf( "Heap pointer 0. pHeapFree, %d\n", pos ); }

		assert ( mHeapFree >= -1 );
	} else {
		hval* pCurr = mHeap + pos;
		hval* pPrev = 0x0;
		hval* pNext = mHeap + mHeapFree;
		
		if ( pCurr == pNext ) {		// Freeing the first block
			mHeapFree = * (hpos*) (pCurr + HEAP_POS);			
			* (hpos*) (mHeap+mHeapFree + HEAP_POS) = 0xFFFFFFFF;			
			* (hpos*) (pCurr + HEAP_POS) = 0xFFFFFFFF;	
			* (hpos*) pCurr = 0xFFFFFFFF;					
			return;		

		}
		
		// Find first block greater than new free pos.
		while ( pNext < pCurr && pNext != mHeap-1 ) {
			pPrev = pNext;
			pNext = mHeap + * (hpos*) (pNext + HEAP_POS);			
		}

		int x = 0;
		if ( pPrev + *(pPrev) == pCurr ) {							// Prev block touches current one (expand previous)
			* (hpos*) pPrev += size;								// - prev.size = prev.size + curr.size
			x=1;
		
		} else if ( pCurr + size == pNext && pNext != mHeap-1 )	{	// Curr block touches next one (expand current block)
			* (hpos*) (pPrev + HEAP_POS) = pos;							// - prev.next = curr
			* (hpos*) (pCurr + HEAP_POS) = * (hpos*) (pNext + HEAP_POS);	// - curr.next = next.next
			* (hpos*) pCurr = size + * (hpos*) pNext;				// - curr.size = size + next.size
			* (hpos*) (pNext) = 0xFFFFFFFF;							// - curr = null
			* (hpos*) (pNext + HEAP_POS) = 0xFFFFFFFF;					// - curr.next = null
			x=2;

		} else {													// Otherwise (linked list insert)
			* (hpos*) (pPrev + HEAP_POS) = pos;							// - prev.next = curr
			if ( pNext != mHeap-1 )
				* (hpos*) (pCurr + HEAP_POS) = (hpos) (pNext - mHeap);		// - curr.next = next				
			else
				* (hpos*) (pCurr + HEAP_POS) = 0xFFFFFFFF;				// - curr.next = null (no next node)
			* (hpos*) pCurr = size;									// - curr.size = size
			x=3;
		}
		if ( pCurr !=mHeap-1 && *(hpos*) (pCurr+HEAP_POS) == 0x0 ) { dbgprintf ( "ERROR: Heap pointer 0. pCurr, %d, %d\n", x, pos );  }
		if ( pPrev !=mHeap-1 && *(hpos*) (pPrev+HEAP_POS) == 0x0 ) { dbgprintf ( "ERROR: Heap pointer 0. pPrev, %d, %d\n", x, pos );  }
		if ( pNext !=mHeap-1 && *(hpos*) (pNext+HEAP_POS) == 0x0 ) { dbgprintf ( "ERROR: Heap pointer 0. pNext, %d, %d\n", x, pos ); }
		if ( *(hpos*) (mHeap+mHeapFree + HEAP_POS) == 0x0 ) { printf ( "ERROR: Heap pointer 0. pHeapFree, %d, %d\n", x, pos );}

		// -- check for bugs (remove eventually)
		pNext = mHeap + mHeapFree;
		while ( pNext != mHeap-1 ) {
			pPrev = pNext;	
			pNext = mHeap + * (hpos*) (pNext + HEAP_POS);
			if ( pNext < pPrev && pNext != mHeap-1 ) {
				printf ( "ERROR: Heap free space out of order. %d, %d\n", x, pos );
			}
		}
		//---
	}
 }


hpos DataX::HeapExpand ( ushort size, ushort& ret  )
{
	mHeapMax *= 2;	
	if ( mHeapMax > HEAP_MAX ) {
		printf ( "Geom heap size exceeds range of index.\n" );		
	}
	hval* pNewHeap = (hval*) malloc ( mHeapMax * sizeof(hval));
	if ( pNewHeap == 0x0 ) {
		printf ( "Geom heap out of memory.\n" );		
	}
	memcpy ( pNewHeap, mHeap, mHeapNum*sizeof(hval) );
	free ( mHeap );
	mHeap = pNewHeap;
	ret = size;
	assert ( mHeapNum >= 0 && mHeapNum < mHeapMax );
	return mHeapNum;
}

hpos DataX::HeapAlloc ( ushort size, ushort& ret  )
{
	hval* pPrev = 0x0;
	hval* pCurr = mHeap + mHeapFree;
	hpos pos = -1;

	if ( mHeapNum + size < mHeapMax ) {
		// Heap not yet full.
		pos = mHeapNum;
		ret = size;
		mHeapNum += size;		
	} else {
		// Heap full, search free space
		if ( mHeapFree == -1 ) {
			pos = HeapExpand ( size, ret );			
			mHeapNum += ret;
		} else {			
			while ( *pCurr < size && pCurr != mHeap-1 ) {
				pPrev = pCurr;
				pCurr = mHeap + * (hpos*) (pCurr + HEAP_POS);
			}
			if ( pCurr != mHeap-1 ) {
				// Found free space.
				if ( pPrev == 0x0 ) {
					mHeapFree = * (hpos*) (pCurr + HEAP_POS);					
					assert ( mHeapFree >= -1 );
				} else {
					* (hpos*) (pPrev+HEAP_POS) = * (hpos*) (pCurr+HEAP_POS);
				}
				pos = (hpos) (pCurr - mHeap);
				ret = *pCurr;
			} else {
				// Heap full, no free space. Expand heap.
				pos = HeapExpand ( size, ret );
				mHeapNum += ret;
			}				
		}		
	}	

	assert ( pos >= 0 && pos <= mHeapNum );
	memset ( mHeap+pos, 0x00, size*sizeof(hval) );
	return pos;
}
