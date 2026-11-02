#include <cholmod.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>

// CPU SYMBOLIC work only. No numerical factorization or covariance runs here.
// Output (simplicial column storage of the supernodal pattern, diagonal first in every column):
//  small int64: [0] perm, [1] cp (n+1), [3] rp (n+1), [5] lp (levels+1), [6] lc (n), [7] off (n+1), [8] sizes (2*levels)
//  large int32: [2] ri (nnz rows), [4] rc (nnz row-structure columns), [9] rx (nnz row-structure positions)
// off[t]: number of off-diagonal entries of columns lc[0..t-1]; a level's work items are [off[lp[l]], off[lp[l+1]]),
// so no per-entry column or work-list array is stored (exact layout change, not an approximation).
struct Symbolic {
    std::vector<int64_t> a[9];
    std::vector<int32_t> b[10];
    int64_t n, nnz, levels;
};
static thread_local std::string last_error;
extern "C" const char* rgbd_symbolic_error() { return last_error.c_str(); }
// The KKT pattern is ordered TOGETHER with the zero-valued query cliques: an ordering computed without the
// cliques (e.g. cuDSS's own nested dissection) separates each point's query columns and explodes the fill.
// first_block > 0: constrained minimum degree (CAMD) eliminating KKT variables [0, first_block) (the whitened
// measurement rows) before all others, so every state/pixel pivot already carries prior + measurement
// information (no "prior minus nearly equal correction" cancellation); first_block == 0: METIS.
extern "C" void* rgbd_symbolic(int64_t n, int64_t nz, const int64_t* row,
    const int64_t* col, int64_t blocks, const int64_t* qp, const int64_t* qi, int64_t first_block) {
    cholmod_common common;
    cholmod_l_start(&common);
    cholmod_triplet* T=nullptr;cholmod_sparse* A=nullptr;cholmod_factor* L=nullptr;
    try {
        common.print=0;common.nmethods=1;common.method[0].ordering=CHOLMOD_METIS;
        common.supernodal=CHOLMOD_SUPERNODAL;common.postorder=1;
        int64_t count=nz;
        for(int64_t b=0;b<blocks;++b) {
            int64_t s=qp[b+1]-qp[b];count+=s*(s+1)/2;
        }
        T=cholmod_l_allocate_triplet(n,n,count,1,CHOLMOD_PATTERN,&common);
        if(!T)throw std::runtime_error("symbolic triplet allocation failed");
        auto tr=(int64_t*)T->i;auto tc=(int64_t*)T->j;int64_t at=0;
        for(int64_t e=0;e<nz;++e) {tr[at]=std::min(row[e],col[e]);tc[at++]=std::max(row[e],col[e]);}
        // Zero-valued symbolic query edges request inverse entries; they never
        // enter the numeric KKT operator and add no statistical information.
        for(int64_t b=0;b<blocks;++b)for(int64_t i=qp[b];i<qp[b+1];++i)
            for(int64_t j=i;j<qp[b+1];++j) {tr[at]=std::min(qi[i],qi[j]);tc[at++]=std::max(qi[i],qi[j]);}
        T->nnz=at;A=cholmod_l_triplet_to_sparse(T,at,&common);
        cholmod_l_free_triplet(&T,&common);
        if(!A)throw std::runtime_error("symbolic sparse conversion failed");
        if(first_block>0) {
            std::vector<int64_t> member(n),perm(n);
            for(int64_t i=0;i<n;++i)member[i]=i<first_block?0:1;
            if(!cholmod_l_camd(A,nullptr,0,member.data(),perm.data(),&common))throw std::runtime_error("CAMD constrained ordering failed");
            common.method[0].ordering=CHOLMOD_GIVEN;
            L=cholmod_l_analyze_p(A,perm.data(),nullptr,0,&common);
        } else L=cholmod_l_analyze(A,&common);
        cholmod_l_free_sparse(&A,&common);
        if(!L || !L->is_super)throw std::runtime_error("CHOLMOD did not produce a supernodal symbolic factor");
        auto result=new Symbolic;result->n=n;
        result->a[0].assign((int64_t*)L->Perm,(int64_t*)L->Perm+n);
        auto sp=(int64_t*)L->super;auto pi=(int64_t*)L->pi;auto s=(int64_t*)L->s;
        auto& cp=result->a[1];cp.resize(n+1);
        int64_t total=0;
        for(size_t b=0;b<L->nsuper;++b)for(int64_t j=sp[b];j<sp[b+1];++j) {
            cp[j]=total;total+=pi[b+1]-pi[b]-(j-sp[b]);
        }
        cp[n]=total;result->nnz=total;
        if(total>=(int64_t(1)<<31)){delete result;throw std::runtime_error("selected-inverse factor needs >= 2^31 entries (int32 storage)");}
        auto& ri=result->b[2];ri.resize(total);
        for(size_t b=0;b<L->nsuper;++b)for(int64_t j=sp[b];j<sp[b+1];++j) {
            int64_t p=cp[j];
            for(int64_t k=pi[b]+j-sp[b];k<pi[b+1];++k) ri[p++]=int32_t(s[k]);
            if(ri[cp[j]]!=j){delete result;throw std::runtime_error("Invalid symbolic diagonal order");}
        }
        cholmod_l_free_factor(&L,&common);
        auto& rp=result->a[3];auto& rc=result->b[4];auto& rx=result->b[9];rp.assign(n+1,0);
        for(auto i:ri)rp[i+1]++;
        for(int64_t j=0;j<n;++j)rp[j+1]+=rp[j];
        rc.resize(total);rx.resize(total);auto next=rp;
        for(int64_t j=0;j<n;++j)for(int64_t p=cp[j];p<cp[j+1];++p) {int64_t q=next[ri[p]]++;rc[q]=int32_t(j);rx[q]=int32_t(p);}
        std::vector<int64_t> level(n,0);
        for(int64_t j=0;j<n;++j)if(cp[j+1]>cp[j]+1) {
            int64_t parent=ri[cp[j]+1];level[parent]=std::max(level[parent],level[j]+1);
        }
        int64_t levels=*std::max_element(level.begin(),level.end())+1;result->levels=levels;
        auto& lp=result->a[5];auto& lc=result->a[6];auto& off=result->a[7];auto& sizes=result->a[8];
        lp.assign(levels+1,0);sizes.assign(levels*2,1);
        for(int64_t j=0;j<n;++j) {int64_t l=level[j];lp[l+1]++;
            sizes[2*l]=std::max(sizes[2*l],rp[j+1]-rp[j]);sizes[2*l+1]=std::max(sizes[2*l+1],cp[j+1]-cp[j]);}
        for(int64_t l=0;l<levels;++l) lp[l+1]+=lp[l];
        lc.resize(n);auto nl=lp;
        for(int64_t j=0;j<n;++j) lc[nl[level[j]]++]=j;
        off.assign(n+1,0);
        for(int64_t t=0;t<n;++t) off[t+1]=off[t]+(cp[lc[t]+1]-cp[lc[t]]-1);
        cholmod_l_finish(&common);
        return result;
    } catch(const std::exception& error) {
        last_error=error.what();if(L)cholmod_l_free_factor(&L,&common);if(A)cholmod_l_free_sparse(&A,&common);
        if(T)cholmod_l_free_triplet(&T,&common);cholmod_l_finish(&common);return nullptr;
    }
}
extern "C" int64_t rgbd_symbolic_size(void* value,int key) {
    auto s=(Symbolic*)value;
    if(key==2||key==4||key==9)return s->b[key].size();
    if(key<9)return s->a[key].size();
    return key==12?s->n:(key==13?s->nnz:(key==14?s->levels:0));
}
extern "C" const int64_t* rgbd_symbolic_array(void* value,int key) {return ((Symbolic*)value)->a[key].data();}
extern "C" const int32_t* rgbd_symbolic_array32(void* value,int key) {return ((Symbolic*)value)->b[key].data();}
extern "C" void rgbd_symbolic_release32(void* value,int key) {std::vector<int32_t>().swap(((Symbolic*)value)->b[key]);}
extern "C" void rgbd_symbolic_free(void* value) {delete (Symbolic*)value;}
