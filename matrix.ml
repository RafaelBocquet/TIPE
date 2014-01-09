
exception Size_mismatch;;

class matrix (width : int) (height : int) = 
  object (self)
  val mutable data : float array = Array.make (width * height) 0.0
  method get i j = data.(i * width + j)
  method set i j (v : float) = data.(i * width + j) <- v
  method iter (f : int -> int -> float -> unit) = 
    for i = 0 to width - 1 do
      for j = 0 to height - 1 do
        f i j (self#get i j)
      done;
    done; ()
  method init f =
    for i = 0 to width - 1 do
      for j = 0 to height - 1 do
        self#set i j (f i j)
      done;
    done; ()
  method width = width
  method height = height
  method add (m : matrix) = 
    if width <> m#width || height <> m#height then raise Size_mismatch
    else
      let r = new matrix width height in
      r#init (fun i j -> (self#get i j) +. (m#get i j)); r
  method multiply (m : matrix) = 
    if width <> m#height then raise Size_mismatch
    else
      let r = new matrix m#width height in
      let rec f i j k = 
        if k = width then 0.0
        else (self#get i k) *. (m#get k j) +. (f i j (k+1))
        in
      r#init (fun i j -> f i j 0); r
  method dump =
    for i = 0 to width - 1 do
      for j = 0 to height - 1 do
        print_float (self#get i j);
        print_string " ";
      done;
      print_newline();
    done; ()
  end;;

let ( *** ) (m1 : matrix) (m2 : matrix) = m1#multiply m2;;
let ( +++ ) (m1 : matrix) (m2 : matrix) = m1#add m2;;

let make w h = new matrix w h;;
