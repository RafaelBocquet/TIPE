
type command = Help | Extract | Archive;;

(*
let a = Matrix.make 2 2 in
a#set 0 0 3;
a#set 1 1 1;
let b = Matrix.make 2 2 in
b#set 0 0 2;
b#set 0 1 2;
b#set 1 0 2;
b#set 1 1 2;
a#dump;
b#dump;
(a#multiply b)#dump;
(b#multiply a)#dump;
(b#add a)#dump;
*)

let rna_hash q = [Queue.top q];; 
let rna_config = [256; 2048; 256], rna_hash;;
let rna = new Rna.rna rna_config;;

(* Commands *)

let print_help () =
  print_string "help, TODO";
  print_newline ();
  ;;

let extract_multiple (filenames : string list) = 
  print_endline "extracting :\n";
  List.iter (fun f -> print_string f; print_newline()) filenames;
  ;;

let archive (filename : string) =
  let file = open_in filename in
    let cont = ref true in
    let by_queue = Queue.create () in
    while !cont do
      try
        let by = input_byte file in
        (* Get RNA prediction + train *)

        let _ = rna#predict by_queue in ();
        rna#train by_queue by;

        (* Add byte to queue *)
        Queue.push by by_queue;
        if Queue.length by_queue > 1 then let _ = Queue.pop by_queue in ();
      with
      | End_of_file -> cont := false;
    done;
    close_in file;
  ;;
let archive_multiple (filenames : string list)  = 
  print_string "archiving :\n";
  List.iter (fun f -> print_string f; print_newline()) filenames;
  ;;

let main () =
  let parse_args () = 
    let mode = ref Help and
      filenames = ref [] in
    let specs =
      [
        ('h', "help", Getopt.set mode Help, None);
        ('x', "extract", Getopt.set mode Extract, None);
        ('a', "archive", Getopt.set mode Archive, None)
      ] in
    Getopt.parse_cmdline specs (fun f -> filenames := f::!filenames);
    (!mode, !filenames) in

  let (mode, filenames) = parse_args () in
    match mode with
    | Help -> print_help ()
    | Extract -> extract_multiple filenames
    | Archive -> archive_multiple filenames
  ;;

main ();;
