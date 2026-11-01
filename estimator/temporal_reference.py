"""Camera-local accepted reference lifecycle; frame age is processing-order age."""
import torch


class ReferenceTable:
    def __init__(self, cameras, points, device, dtype):
        self.frame = torch.full((cameras,points),-1,dtype=torch.long,device=device)
        self.ordinal = torch.full_like(self.frame,-1)
        self.pixel = torch.full_like(self.frame,-1)
        self.state = torch.zeros((cameras,points,6),dtype=dtype,device=device)
        self.source_identity = None

    def initialize_from_source_catalogue(self, source, k, state):
        p = torch.arange(len(state),device=state.device)
        c = torch.as_tensor(source[:,1],device=state.device,dtype=torch.long)
        pixels = torch.as_tensor(source[:,2],device=state.device,dtype=torch.long)
        self.frame[c,p] = k; self.ordinal[c,p] = 0
        self.pixel[c,p] = pixels; self.state[c,p] = state
        self.source_identity = torch.stack((c,p,pixels),-1).clone()

    def commit_accepted_references(self, owners, frame, state, ordinal):
        c,y,x = torch.where(owners.owner_id>=0)
        p = owners.owner_id[c,y,x]
        ok = frame.mask[c,y,x] & frame.raw_valid[c,y,x] & torch.isfinite(frame.rgb[c,y,x]).all(1) & torch.isfinite(frame.depth[c,y,x]) & (frame.depth[c,y,x]>0)
        c,p,y,x = c[ok],p[ok],y[ok],x[ok]
        self.frame[c,p] = frame.k; self.ordinal[c,p] = ordinal
        self.pixel[c,p] = y*frame.W+x; self.state[c,p] = state[p]
        return int(len(p))
