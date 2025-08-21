import argparse, yaml, os, hashlib, sys

def sha256_file(path, chunk=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def resolve_manifest_yaml(data_cfg_path):
    with open(data_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["manifest_yaml"]

def verify_manifest(manifest_yaml, write_missing=False):
    with open(manifest_yaml, "r") as f:
        man = yaml.safe_load(f)
    changed = False
    ok = True
    for item in man.get("files", []):
        p = item["path"]
        exp = item.get("sha256")
        if not os.path.exists(p):
            print(f"[verify_manifests] MISSING FILE: {p}", file=sys.stderr)
            ok = False
            continue
        got = sha256_file(p)
        if exp in (None, "FILL_ME", "", "null"):
            if write_missing:
                item["sha256"] = got
                changed = True
                print(f"[verify_manifests] wrote sha256 for {p}")
            else:
                print(f"[verify_manifests] missing sha256 for {p} (got {got})")
                ok = False
        else:
            if got != exp:
                print(f"[verify_manifests] HASH MISMATCH for {p}: expected {exp}, got {got}", file=sys.stderr)
                ok = False
    if changed and write_missing:
        with open(manifest_yaml, "w") as f:
            yaml.safe_dump(man, f, sort_keys=False)
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write-missing", action="store_true")
    ap.add_argument("data_cfgs", nargs="+")
    args = ap.parse_args()
    if args.verify and args.write_missing:
        print("Use either --verify or --write-missing, not both.", file=sys.stderr)
        sys.exit(2)
    overall_ok = True
    for cfg in args.data_cfgs:
        man_yaml = resolve_manifest_yaml(cfg)
        ok = verify_manifest(man_yaml, write_missing=args.write_missing)
        overall_ok = overall_ok and ok
    if args.verify and not overall_ok:
        sys.exit(3)

if __name__ == "__main__":
    main()
