type rna_config = {
    layer_config : int list;
    hash_function : int Queue.t -> int list
};;

let activation_function (d : float) : float = 1.0 /. (1.0 +. exp d);;
let activation_derivative (d : float) : float = d;;

class rna (rna_config : rna_config) =
  object (self)
  
  val layers = Array.init (List.length rna_config.layer_config - 1) (fun i -> new Matrix.matrix (List.nth rna_config.layer_config i) (List.nth rna_config.layer_config (i + 1)))
  
  val in_size = List.nth rna_config.layer_config 0
  
  val out_size = List.nth rna_config.layer_config (List.length rna_config.layer_config - 1)
  
  method init =
    let seed = 951628473 in
    Random.init seed;
    Array.iter (fun m -> m#init (fun _ _ -> Random.float 1.0)) layers; ()
  
  method train (input : int Queue.t) (except : int) = 0
  
  method predict (input : int Queue.t) =
    let hash_list = rna_config.hash_function input in
    let in_mat = Matrix.make 1 in_size in
    List.iter (fun i -> in_mat#set 1 (i mod in_size) 1.0) hash_list;
    let activate vec = vec#transform (fun _ _ x -> activation_function x) in
    let rec get_out in_vec i =
      if i = Array.length layers then in_vec
      else let tmp_out = layers.(i)#multiply in_vec in
        activate tmp_out;
        get_out tmp_out (i + 1)
      in get_out in_mat 0
  end;;