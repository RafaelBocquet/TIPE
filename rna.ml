type rna_config = (int list) * (int Queue.t -> int list);;

class rna (layer_config, hash : rna_config) =
  object (self)
  val layers = Array.init (List.length layer_config - 1) (fun i -> new Matrix.matrix (List.nth layer_config i) (List.nth layer_config (i + 1)))
  val in_size = List.nth layer_config 0
  val out_size = List.nth layer_config (List.length layer_config - 1)
  method train (input : int Queue.t) (except : int) = 0
  method predict (input : int Queue.t) =
    let hash_list = hash input in
    let in_mat = Matrix.make 1 in_size in
    List.iter (fun i -> in_mat#set 1 i 1.0) hash_list;
  end;;