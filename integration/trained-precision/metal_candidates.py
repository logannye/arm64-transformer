"""Emit an explicit four-policy Metal development inventory, without execution.

Policy bits: (1) normalization reciprocal/affine intermediates as two-float
expansions; (2) compensated attention-value contraction. FP32 storage persists.
This inventory targets the reference shader path, not optimized policy kernels.
"""

import argparse
import hashlib
import json
from pathlib import Path

NORM_OLD = (
    """    float scale=1.0f/sqrt(variance.x+variance.y);
    inv[r]=scale;
    for(uint c=0;c<a.w;++c){float2 normalized=ff_scale(ff_add(float2(x[r*a.w+c],0),-mean),scale);
        float z=normalized.x+normalized.y;norm[r*a.w+c]=z;
"""
    "        float2 affine=ff_add(ff_product(z,gamma[c]),float2(beta[c],0));"
    "y[r*a.w+c]=affine.x+affine.y;}"
)
NORM_NEW = """    // Recover the low part of the model's exact 1e-5 epsilon.
    if(a.epsilon==1.0e-5f) variance=ff_add(variance,float2(0,2.5262125e-13f));
    float approximate=1.0f/sqrt(variance.x+variance.y);
    float2 residual=ff_add(float2(1,0),-ff_multiply(variance,ff_product(approximate,approximate)));
    float correction=(0.5f*approximate)*(residual.x+residual.y);
    float2 scale=ff_add(float2(approximate,0),float2(correction,0));
    inv[r]=scale.x+scale.y;
    for(uint c=0;c<a.w;++c){float2 normalized=ff_multiply(ff_add(float2(x[r*a.w+c],0),-mean),scale);
        norm[r*a.w+c]=normalized.x+normalized.y;
        float2 affine=ff_add(ff_scale(normalized,gamma[c]),float2(beta[c],0));
        y[r*a.w+c]=affine.x+affine.y;}"""
VALUES_OLD = """    uint c=i%a.w,q=(i/a.w)%a.t,b=i/(a.w*a.t),h=c/(a.w/a.h);float s=0;
    for(uint j=0;j<=q;++j)s=fma(p[((b*a.h+h)*a.t+q)*a.t+j],qkv[(b*a.t+j)*3*a.w+2*a.w+c],s);
    out[i]=s;"""
VALUES_NEW = """    uint c=i%a.w,q=(i/a.w)%a.t,b=i/(a.w*a.t),h=c/(a.w/a.h);float2 s=0;
    for(uint j=0;j<=q;++j)s=ff_add(s,ff_product(
        p[((b*a.h+h)*a.t+q)*a.t+j],qkv[(b*a.t+j)*3*a.w+2*a.w+c]));
    out[i]=s.x+s.y;"""


def generate(source, source_sha256, output):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != source_sha256:
        raise ValueError("Unbound baseline shader")
    text = data.decode()
    if text.count(NORM_OLD) != 1 or text.count(VALUES_OLD) != 1:
        raise ValueError("Baseline operation body changed")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "trained-metal-precision-candidates-v1",
        "baseline_shader_sha256": source_sha256,
        "formal_eligible": False,
        "compiled": False,
        "scope": "reference shader path; optimized policy paths are unchanged",
        "policies": {},
    }
    for policy in range(4):
        candidate = text
        if policy & 1:
            candidate = candidate.replace(NORM_OLD, NORM_NEW)
        if policy & 2:
            candidate = candidate.replace(VALUES_OLD, VALUES_NEW)
        path = output / f"policy-{policy}.metal"
        path.write_text(candidate)
        manifest["policies"][str(policy)] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.source_sha256, args.output), indent=2))
