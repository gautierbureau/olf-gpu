// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The exported name of a native method of class cls in package com.powsybl.openloadflow.<pkg> (dots as
// underscores): the JNI mangling in one place, so moving natives to a facade renames their symbols in one token.
#ifndef OLF_GPU_JNI_OLF_JNI_H
#define OLF_GPU_JNI_OLF_JNI_H

#define OLF_JNI(pkg, cls, m) Java_com_powsybl_openloadflow_##pkg##_##cls##_##m

#endif  // OLF_GPU_JNI_OLF_JNI_H
