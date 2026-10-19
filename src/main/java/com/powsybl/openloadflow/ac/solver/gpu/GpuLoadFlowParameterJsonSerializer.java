/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.powsybl.commons.extensions.ExtensionJsonSerializer;
import com.powsybl.commons.json.JsonUtil;
import com.powsybl.loadflow.LoadFlowParameters;

import java.io.IOException;

/**
 * The JSON form of {@link GpuLoadFlowParameters} inside a load flow parameters file, as
 * {@code OpenLoadFlowParameterJsonSerializer} does for OLF's own parameters. Each field is written under its
 * parameter name.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuLoadFlowParameterJsonSerializer implements ExtensionJsonSerializer<LoadFlowParameters, GpuLoadFlowParameters> {

    @Override
    public String getExtensionName() {
        return GpuLoadFlowParameters.NAME;
    }

    @Override
    public String getCategoryName() {
        return "loadflow-parameters";
    }

    @Override
    public Class<? super GpuLoadFlowParameters> getExtensionClass() {
        return GpuLoadFlowParameters.class;
    }

    /**
     * Specifies serialization for our extension: ignore name and extendable
     */
    private interface SerializationSpec {

        @JsonIgnore
        String getName();

        @JsonIgnore
        LoadFlowParameters getExtendable();
    }

    private static ObjectMapper createMapper() {
        return JsonUtil.createObjectMapper()
                .addMixIn(GpuLoadFlowParameters.class, SerializationSpec.class);
    }

    @Override
    public void serialize(GpuLoadFlowParameters extension, JsonGenerator jsonGenerator, SerializerProvider serializerProvider) throws IOException {
        createMapper().writeValue(jsonGenerator, extension);
    }

    @Override
    public GpuLoadFlowParameters deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException {
        return createMapper().readValue(jsonParser, GpuLoadFlowParameters.class);
    }

    @Override
    public GpuLoadFlowParameters deserializeAndUpdate(JsonParser jsonParser, DeserializationContext deserializationContext,
                                                      GpuLoadFlowParameters extension) throws IOException {
        return createMapper().readerForUpdating(extension).readValue(jsonParser, GpuLoadFlowParameters.class);
    }
}
