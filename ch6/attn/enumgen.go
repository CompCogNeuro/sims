// Code generated by "core generate -add-types"; DO NOT EDIT.

package main

import (
	"cogentcore.org/core/enums"
)

var _TestTypeValues = []TestType{0, 1, 2, 3, 4}

// TestTypeN is the highest valid value for type TestType, plus one.
const TestTypeN TestType = 5

var _TestTypeValueMap = map[string]TestType{`MultiObjs`: 0, `StdPosner`: 1, `ClosePosner`: 2, `ReversePosner`: 3, `ObjAttn`: 4}

var _TestTypeDescMap = map[TestType]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``}

var _TestTypeMap = map[TestType]string{0: `MultiObjs`, 1: `StdPosner`, 2: `ClosePosner`, 3: `ReversePosner`, 4: `ObjAttn`}

// String returns the string representation of this TestType value.
func (i TestType) String() string { return enums.String(i, _TestTypeMap) }

// SetString sets the TestType value from its string representation,
// and returns an error if the string is invalid.
func (i *TestType) SetString(s string) error {
	return enums.SetString(i, s, _TestTypeValueMap, "TestType")
}

// Int64 returns the TestType value as an int64.
func (i TestType) Int64() int64 { return int64(i) }

// SetInt64 sets the TestType value from an int64.
func (i *TestType) SetInt64(in int64) { *i = TestType(in) }

// Desc returns the description of the TestType value.
func (i TestType) Desc() string { return enums.Desc(i, _TestTypeDescMap) }

// TestTypeValues returns all possible values for the type TestType.
func TestTypeValues() []TestType { return _TestTypeValues }

// Values returns all possible values for the type TestType.
func (i TestType) Values() []enums.Enum { return enums.Values(_TestTypeValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i TestType) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *TestType) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "TestType") }

var _LesionTypeValues = []LesionType{0, 1, 2, 3}

// LesionTypeN is the highest valid value for type LesionType, plus one.
const LesionTypeN LesionType = 4

var _LesionTypeValueMap = map[string]LesionType{`NoLesion`: 0, `LesionSpat1`: 1, `LesionSpat2`: 2, `LesionSpat12`: 3}

var _LesionTypeDescMap = map[LesionType]string{0: ``, 1: ``, 2: ``, 3: ``}

var _LesionTypeMap = map[LesionType]string{0: `NoLesion`, 1: `LesionSpat1`, 2: `LesionSpat2`, 3: `LesionSpat12`}

// String returns the string representation of this LesionType value.
func (i LesionType) String() string { return enums.String(i, _LesionTypeMap) }

// SetString sets the LesionType value from its string representation,
// and returns an error if the string is invalid.
func (i *LesionType) SetString(s string) error {
	return enums.SetString(i, s, _LesionTypeValueMap, "LesionType")
}

// Int64 returns the LesionType value as an int64.
func (i LesionType) Int64() int64 { return int64(i) }

// SetInt64 sets the LesionType value from an int64.
func (i *LesionType) SetInt64(in int64) { *i = LesionType(in) }

// Desc returns the description of the LesionType value.
func (i LesionType) Desc() string { return enums.Desc(i, _LesionTypeDescMap) }

// LesionTypeValues returns all possible values for the type LesionType.
func LesionTypeValues() []LesionType { return _LesionTypeValues }

// Values returns all possible values for the type LesionType.
func (i LesionType) Values() []enums.Enum { return enums.Values(_LesionTypeValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i LesionType) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *LesionType) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "LesionType")
}

var _LesionSizeValues = []LesionSize{0, 1}

// LesionSizeN is the highest valid value for type LesionSize, plus one.
const LesionSizeN LesionSize = 2

var _LesionSizeValueMap = map[string]LesionSize{`LesionHalf`: 0, `LesionFull`: 1}

var _LesionSizeDescMap = map[LesionSize]string{0: ``, 1: ``}

var _LesionSizeMap = map[LesionSize]string{0: `LesionHalf`, 1: `LesionFull`}

// String returns the string representation of this LesionSize value.
func (i LesionSize) String() string { return enums.String(i, _LesionSizeMap) }

// SetString sets the LesionSize value from its string representation,
// and returns an error if the string is invalid.
func (i *LesionSize) SetString(s string) error {
	return enums.SetString(i, s, _LesionSizeValueMap, "LesionSize")
}

// Int64 returns the LesionSize value as an int64.
func (i LesionSize) Int64() int64 { return int64(i) }

// SetInt64 sets the LesionSize value from an int64.
func (i *LesionSize) SetInt64(in int64) { *i = LesionSize(in) }

// Desc returns the description of the LesionSize value.
func (i LesionSize) Desc() string { return enums.Desc(i, _LesionSizeDescMap) }

// LesionSizeValues returns all possible values for the type LesionSize.
func LesionSizeValues() []LesionSize { return _LesionSizeValues }

// Values returns all possible values for the type LesionSize.
func (i LesionSize) Values() []enums.Enum { return enums.Values(_LesionSizeValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i LesionSize) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *LesionSize) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "LesionSize")
}
