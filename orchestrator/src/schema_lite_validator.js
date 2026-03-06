function describeType(value) {
  if (Array.isArray(value)) return "array";
  if (value === null) return "null";
  return typeof value;
}

function joinPath(base, key) {
  return base ? `${base}.${key}` : key;
}

export function validateJsonSchemaLite(schema, value, path = "") {
  const errors = [];
  const expectedType = schema?.type;

  if (expectedType === "object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return [`${path || "$"} expected object, got ${describeType(value)}`];
    }
    const required = Array.isArray(schema.required) ? schema.required : [];
    for (const key of required) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) {
        errors.push(`${joinPath(path || "$", key)} missing`);
      }
    }
    const properties = schema.properties || {};
    for (const [key, subSchema] of Object.entries(properties)) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) continue;
      errors.push(...validateJsonSchemaLite(subSchema, value[key], joinPath(path || "$", key)));
    }
    return errors;
  }

  if (expectedType === "array") {
    if (!Array.isArray(value)) {
      return [`${path || "$"} expected array, got ${describeType(value)}`];
    }
    if (Number.isFinite(schema.minItems) && value.length < schema.minItems) {
      errors.push(`${path || "$"} requires at least ${schema.minItems} items`);
    }
    if (schema.items) {
      value.forEach((item, index) => {
        errors.push(...validateJsonSchemaLite(schema.items, item, `${path || "$"}[${index}]`));
      });
    }
    return errors;
  }

  if (expectedType === "string") {
    if (typeof value !== "string") {
      return [`${path || "$"} expected string, got ${describeType(value)}`];
    }
    if (Number.isFinite(schema.minLength) && value.length < schema.minLength) {
      errors.push(`${path || "$"} requires minLength ${schema.minLength}`);
    }
    return errors;
  }

  if (expectedType === "boolean") {
    if (typeof value !== "boolean") {
      return [`${path || "$"} expected boolean, got ${describeType(value)}`];
    }
    return errors;
  }

  return errors;
}
