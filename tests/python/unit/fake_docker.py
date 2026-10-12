#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for the docker CLI used by isolate_cpus.py tests.

State: $FAKE_DOCKER_STATE = {"containers": [{"id", "name", "cpus", "mems",
"pid", "cgroup"}]}; ``update --cpuset-cpus`` also writes the container's
cpuset.cpus in the fake cgroup root ($FAKE_CGROUP_ROOT)."""

import json
import os
import sys

path = os.environ["FAKE_DOCKER_STATE"]
st = json.load(open(path))
args = sys.argv[1:]
cmd = args[0]
if cmd == "ps":
    for c in st["containers"]:
        print(c["id"])
elif cmd == "inspect":
    ids = args[3:]
    for c in st["containers"]:
        if c["id"] in ids:
            print(
                "\t".join(
                    [
                        c["id"],
                        "/" + c["name"],
                        c["cpus"],
                        c["mems"],
                        str(c["pid"]),
                    ]
                )
            )
elif cmd == "update":
    cpus = args[1].split("=", 1)[1]
    cid = args[2]
    for c in st["containers"]:
        if c["id"] == cid:
            c["cpus"] = cpus
            root = os.environ.get("FAKE_CGROUP_ROOT")
            if root and c.get("cgroup"):
                with open(
                    os.path.join(root, c["cgroup"].strip("/"), "cpuset.cpus"),
                    "w",
                ) as f:
                    f.write(cpus)
            st.setdefault("updates", []).append([c["name"], cpus])
    json.dump(st, open(path, "w"))
elif cmd == "exec":
    # exec [-e K=V]... NAME CMD...: run CMD here (our container = this machine)
    rest, env = args[1:], dict(os.environ)
    while rest and rest[0] == "-e":
        k, _, v = rest[1].partition("=")
        env[k] = v
        rest = rest[2:]
    os.execvpe(rest[1], rest[1:], env)
else:
    sys.exit("fake docker: unsupported " + " ".join(args))
