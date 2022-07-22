// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error @+1 {{expected 'luminous.container_module' attribute to be attached to 'builtin.module'}}
func.func @wrong () attributes {luminous.container_module}  {
  luminous.module @capsules {
  }
  return
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{custom op 'luminous.func' requires named arguments}}
    luminous.func @capsule (f32) {
      luminous.return
    }
  }
  func.func @m() {
    luminous.dispatch @non-existent::@non-existent()
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{custom op 'luminous.func' requires named arguments}}
    luminous.func @capsule (f32) {
      luminous.return
    }
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{'luminous.func' op attribute 'function_type' failed to satisfy constraint: type attribute of function type}}
    "luminous.func"() ({
      luminous.return
    }) {sym_name="capsule", function_type = f32} : () -> ()
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{custom op 'luminous.func' expected no return type}}
    luminous.func @capsule (%a: f32) -> index {
      luminous.return
    }
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{op expected 1 arguments to body region}}
    "luminous.func"() ( {
    ^bb0(%arg0: f32, %arg1: f32):
      "luminous.return"() : () -> ()
    } ) {sym_name = "capsule", function_type = (f32) -> ()} : () -> ()
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{op expected 2 arguments to body region}}
    "luminous.func"() ( {
    ^bb0(%arg0: f32):
      "luminous.return"() : () -> ()
    } ) {sym_name = "capsule", function_type = (f32, f32) -> ()} : () -> ()
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    // expected-error @+1 {{op expected body region argument #1 to be of type 'f32', got 'i32'}}
    "luminous.func"() ( {
    ^bb0(%arg0: f32, %arg1: i32):
      "luminous.return"() : () -> ()
    } ) {sym_name = "capsule", function_type = (f32, f32) -> ()} : () -> ()
  }
}

// -----

module attributes {luminous.container_module}  {
  luminous.module @capsules {
    llvm.mlir.global internal constant @capsule("capsule\00")
  }

  func.func @m() {
    // expected-error @+1 {{'luminous.dispatch' op symbol reference attribute 'function' must be specified}}
    %0 = "luminous.dispatch"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
  }
}

// -----

module attributes {luminous.container_module} {
  func.func @dispatch_undefined_module(%sz: index) {
    // expected-error@+1 {{luminous.module 'capsules' is undefined}}
    %t0 = luminous.dispatch @capsules::@capsule(%sz: index, %sz: index)
    return
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
  }

  func.func @dispatch_missing_module_attribute(%sz: index) {
    // expected-error@+1 {{'luminous.dispatch' op capsule function '@capsules::@capsule' is undefined}}
    %t0 = luminous.dispatch @capsules::@capsule(%sz: index)
    return
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    luminous.func @capsule(%sz: index) {
      luminous.return
    }
  }

  func.func @dispatch_capsule_operand_size(%sz: index, %arg: f32) {
    // expected-error@+1 {{got 2 capsule function operands but expected 1}}
    %t0 = luminous.dispatch @capsules::@capsule(%sz: index, %arg: f32)
    return
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    luminous.func @capsule(%sz: index) {
      luminous.return
    }
  }

  func.func @dispatch_capsule_operand_types(%arg: f32) {
    // expected-error@+1 {{'luminous.dispatch' op type of function argument 0 does not match}}
    %t0 = luminous.dispatch @capsules::@capsule(%arg: f32)
    return
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    luminous.func @capsule(%sz: index) {
      luminous.return
    }
  }

  func.func @dispatch_no_result(%sz: index) {
    // expected-error@+1 {{custom op 'luminous.dispatch' must have a result}}
    luminous.dispatch @capsules::@capsule(%sz: index)
    return
  }
}

// -----

module {
  luminous.module @capsules {
    luminous.func @capsule(%sz: index) {
      luminous.return
    }
  }

  func.func @dispatch_no_result(%sz: index) {
    // expected-error@+1 {{'luminous.dispatch' op expected the closest surrounding module to have the 'luminous.container_module' attribute}}
    %t0 = luminous.dispatch @capsules::@capsule(%sz: index)
    return
  }
}

// -----

module attributes {luminous.container_module} {
  luminous.module @capsules {
    luminous.func @capsule(%sz: index) {
      luminous.return
    }
  }

  module {
    func.func @dispatch_overly_nested(%sz: index) {
      // expected-error@+1 {{'luminous.dispatch' op expected the closest surrounding module to have the 'luminous.container_module' attribute}}
      %t0 = luminous.dispatch @capsules::@capsule(%sz: index)
      return
    }
  }
}

// -----

module attributes {luminous.container_module} {
  func.func @launch() {
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    // expected-error@+1 {{'luminous.launch' op shape and step variables must be the same size.}}
    luminous.launch shape (%c1024) step (%c64, %c1) {
    ^bb0(%arg0: index, %arg1: index):
      luminous.yield
    }
    return
  }
}