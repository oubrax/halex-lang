# basics
```rs
let x = 2 // type inference x is i32
fn y() { 1 } // last statement = return value
```
```rs
use module.{a, b, c};
use module.*;

fn add(a: *i32) {
    *a = 2
}

let num = 1 // variables are mutable

add(&num)
&num |> add // works too

```

