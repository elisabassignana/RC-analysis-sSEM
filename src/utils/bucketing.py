import os
import json
from dotenv import load_dotenv

load_dotenv()

class Buckets():

	def __init__(self, attribute, domain, labels2id, train_path= None, multi_label=False, double_markers=False):

		self.attribute  = attribute
		self.double_markers = double_markers

		if attribute == 'none':
			self.sentences = {('all'): []}
			self.entities_1 = {('all'): []}
			self.entities_2 = {('all'): []}
			self.relations = {('all'): []}
			if self.double_markers:
				self.entities_end1 = {('all'): []}
				self.entities_end2 = {('all'): []}

			self.bucket_values = {('all'): []}

		else:
			if self.attribute in os.getenv("categorical").split():
				self.attribute_name = self.attribute
			else:
				self.attribute_name = f"{domain}-{self.attribute}"
			self.buckets = self.env_to_buckets()
			self.sentences = {bucket: [] for bucket in self.buckets}
			self.entities_1 = {bucket: [] for bucket in self.buckets}
			self.entities_2 = {bucket: [] for bucket in self.buckets}
			self.relations = {bucket: [] for bucket in self.buckets}
			if self.double_markers:
				self.entities_end1 = {bucket: [] for bucket in self.buckets}
				self.entities_end2 = {bucket: [] for bucket in self.buckets}

			self.bucket_values = {bucket: [] for bucket in self.buckets}

		if train_path != None:
			self.get_train_set(train_path)
		self.multi_label = multi_label
		self.labels2id = labels2id

		self.entity_type_map = {
			"academicjournal": "misc",
			"album": "misc",
			"algorithm": "misc",
			"astronomicalobject": "misc",
			"award": "misc",
			"band": "organisation",
			"book": "misc",
			"chemicalcompound": "misc",
			"chemicalelement": "misc",
			"conference": "event",
			"country": "location",
			"discipline": "misc",
			"election": "event",
			"enzyme": "misc",
			"event": "event",
			"field": "misc",
			"literarygenre": "misc",
			"location": "location",
			"magazine": "misc",
			"metrics": "misc",
			"misc": "misc",
			"musicalartist": "person",
			"musicalinstrument": "misc",
			"musicgenre": "misc",
			"organisation": "organisation",
			"person": "person",
			"poem": "misc",
			"politicalparty": "organisation",
			"politician": "person",
			"product": "misc",
			"programlang": "misc",
			"protein": "misc",
			"researcher": "person",
			"scientist": "person",
			"song": "misc",
			"task": "misc",
			"theory": "misc",
			"university": "organisation",
			"writer": "person"
		}

	def env_to_buckets(self):

		buckets = []
		for elem in json.loads(os.environ[self.attribute_name]):
			buckets.append(tuple(elem.split(' ')))
		return buckets

	def set_json_instance(self, json_instance):
		self.instance = json_instance

	def compute_current_attribute(self, entity_pair, relation_types):

		if self.attribute == 'none':
			self.current = 'all'

		elif self.attribute == 'sentence_length':
			self.current = len(self.instance["sentence"])

		elif self.attribute == 'entity_density':
			self.current = len(self.instance["ner"]) * 100 / len(self.instance["sentence"])

		elif self.attribute == 'entity_pair_density':
			self.current = len(self.instance["relations"]) * 100 / len(self.instance["sentence"])

		elif self.attribute == 'oov_token_density':
			tot_oov = 0
			for token in self.instance["sentence"]:
				if token.lower() not in self.tokens_train:
					tot_oov += 1
			self.current = tot_oov * 100 / len(self.instance["sentence"])

		elif self.attribute == 'entity_distance':
			self.current = abs(entity_pair[0][1] - entity_pair[1][0]) - 1

		elif self.attribute == 'entity_length':
			self.current = (entity_pair[0][1] - entity_pair[0][0] + 1) + (entity_pair[1][1] - entity_pair[1][0] + 1)

		elif self.attribute == 'entity_type' or self.attribute == 'entity_type_frequency':
			# self.current is a list
			self.current = [self.entity_type_map[entity_pair[0][2]], self.entity_type_map[entity_pair[1][2]]]

		elif self.attribute == 'relation_type_frequency':
			# self.current is a list
			self.current = relation_types

		elif self.attribute == 'iv_entities':
			# retrieve entity tokens
			entity_tokens_1 = ''
			for i in range(entity_pair[0][0], entity_pair[0][1] + 1):
				entity_tokens_1 += f'{self.instance["sentence"][i].lower()} '
			entity_tokens_2 = ''
			for i in range(entity_pair[1][0], entity_pair[1][1] + 1):
				entity_tokens_2 += f'{self.instance["sentence"][i].lower()} '
			# update OOV entities
			if entity_tokens_1.strip() in self.entities_train and entity_tokens_2.strip() in self.entities_train:
				self.current = 2
			elif entity_tokens_1.strip() in self.entities_train or entity_tokens_2.strip() in self.entities_train:
				self.current = 1
			else:
				self.current = 0

	def get_train_set(self, train_path):

		self.entities_train = set()
		self.entity_pairs_train = set()
		self.tokens_train = set()

		with open(train_path) as train:
			for line in train:
				instance = json.loads(line)

				# add entities
				for ent in instance["ner"]:
					entity_tokens = ''
					for i in range(ent[0], ent[1] + 1):
						entity_tokens += f'{instance["sentence"][i].lower()} '
					self.entities_train.add(entity_tokens.strip())

				# add relations
				for rel in instance["relations"]:
					entity_tokens_1 = ''
					for i in range(rel[0], rel[1] + 1):
						entity_tokens_1 += f'{instance["sentence"][i].lower()} '
					entity_tokens_2 = ''
					for i in range(rel[2], rel[3] + 1):
						entity_tokens_2 += f'{instance["sentence"][i].lower()} '
					self.entity_pairs_train.add((entity_tokens_1.strip(), entity_tokens_2.strip()))

			# add tokens
			for token in instance["sentence"]:
				self.tokens_train.add(token.lower())

	def update_buckets(self, sentence, ent1, ent2, relations):

		# before updating self.current save the attribute value of the instance
		self.current_value = self.current

		# handle upper bound
		if self.attribute == 'entity_length' or self.attribute == 'entity_distance' or self.attribute == 'sentence_length':
			max_value = list(self.sentences.keys())[-1][-1][:-1]
			if self.current >= int(max_value):
				self.current = list(self.sentences.keys())[-1][-1]

		# handle density
		if self.attribute == 'entity_density' or self.attribute == 'entity_pair_density' or self.attribute == 'oov_token_density':
			if self.current >= 0 and self.current <= 5:
				self.current = '0-5%'
			elif self.current > 5 and self.current <= 10:
				self.current = '5-10%'
			elif self.current > 10 and self.current <= 15:
				self.current = '10-15%'
			elif self.current > 15 and self.current <= 20:
				self.current = '15-20%'
			elif self.current > 20 and self.current <= 25:
				self.current = '20-25%'
			elif self.current > 25 and self.current <= 30:
				self.current = '25-30%'
			elif self.current > 30 and self.current <= 35:
				self.current = '30-35%'
			elif self.current > 35 and self.current <= 40:
				self.current = '35-40%'
			elif self.current > 40 and self.current <= 45:
				self.current = '40-45%'
			elif self.current > 45 and self.current <= 50:
				self.current = '45-50%'
			elif self.current > 50 and self.current <= 55:
				self.current = '50-55%'
			elif self.current > 55 and self.current <= 60:
				self.current = '55-60%'
			elif self.current > 60 and self.current <= 65:
				self.current = '60-65%'
			elif self.current > 65 and self.current <= 70:
				self.current = '65-70%'
			elif self.current > 70 and self.current <= 75:
				self.current = '70-75%'
			elif self.current > 75 and self.current <= 80:
				self.current = '75-80%'
			elif self.current > 80 and self.current <= 85:
				self.current = '80-85%'
			elif self.current > 85 and self.current <= 90:
				self.current = '85-90%'
			elif self.current > 90 and self.current <= 95:
				self.current = '90-95%'
			elif self.current > 95 and self.current <= 100:
				self.current = '95-100%'

		# normalize all to list
		if type(self.current) != list:
			self.current = [self.current]

		for i in range(len(self.current)): # for entity and relation type there are multiple "current"
			for bucket in self.sentences.keys():
				if str(self.current[i]) in bucket:
					# update sentences, entities and gold relations with the current instance
					self.sentences[bucket].append(sentence)
					self.entities_1[bucket].append(ent1)
					self.entities_2[bucket].append(ent2)
					if self.multi_label:
						instance_labels = [0] * len(self.labels2id.keys())
						for rel in relations:
							instance_labels[self.labels2id[rel]] = 1
						self.relations[bucket].append(instance_labels)
					else:
						self.relations[bucket].append(self.labels2id[relations[0]])

					# update attribute value
					self.bucket_values[bucket].append(self.current_value)

					break

	def update_buckets_double_markers(self, sentence, ent1, ent2, relations, ent1end, ent2end):

		# before updating self.current save the attribute value of the instance
		self.current_value = self.current

		# handle upper bound
		if self.attribute == 'entity_length' or self.attribute == 'entity_distance' or self.attribute == 'sentence_length':
			max_value = list(self.sentences.keys())[-1][-1][:-1]
			if self.current >= int(max_value):
				self.current = list(self.sentences.keys())[-1][-1]

		# handle density
		if self.attribute == 'entity_density' or self.attribute == 'entity_pair_density' or self.attribute == 'oov_token_density':
			if self.current >= 0 and self.current <= 5:
				self.current = '0-5%'
			elif self.current > 5 and self.current <= 10:
				self.current = '5-10%'
			elif self.current > 10 and self.current <= 15:
				self.current = '10-15%'
			elif self.current > 15 and self.current <= 20:
				self.current = '15-20%'
			elif self.current > 20 and self.current <= 25:
				self.current = '20-25%'
			elif self.current > 25 and self.current <= 30:
				self.current = '25-30%'
			elif self.current > 30 and self.current <= 35:
				self.current = '30-35%'
			elif self.current > 35 and self.current <= 40:
				self.current = '35-40%'
			elif self.current > 40 and self.current <= 45:
				self.current = '40-45%'
			elif self.current > 45 and self.current <= 50:
				self.current = '45-50%'
			elif self.current > 50 and self.current <= 55:
				self.current = '50-55%'
			elif self.current > 55 and self.current <= 60:
				self.current = '55-60%'
			elif self.current > 60 and self.current <= 65:
				self.current = '60-65%'
			elif self.current > 65 and self.current <= 70:
				self.current = '65-70%'
			elif self.current > 70 and self.current <= 75:
				self.current = '70-75%'
			elif self.current > 75 and self.current <= 80:
				self.current = '75-80%'
			elif self.current > 80 and self.current <= 85:
				self.current = '80-85%'
			elif self.current > 85 and self.current <= 90:
				self.current = '85-90%'
			elif self.current > 90 and self.current <= 95:
				self.current = '90-95%'
			elif self.current > 95 and self.current <= 100:
				self.current = '95-100%'

		# normalize all to list
		if type(self.current) != list:
			self.current = [self.current]

		for i in range(len(self.current)): # for entity and relation type there are multiple "current"
			for bucket in self.sentences.keys():
				if str(self.current[i]) in bucket:
					# update sentences, entities and gold relations with the current instance
					self.sentences[bucket].append(sentence)
					self.entities_1[bucket].append(ent1)
					self.entities_2[bucket].append(ent2)
					self.entities_end1[bucket].append(ent1end)
					self.entities_end2[bucket].append(ent2end)
					if self.multi_label:
						instance_labels = [0] * len(self.labels2id.keys())
						for rel in relations:
							instance_labels[self.labels2id[rel]] = 1
						self.relations[bucket].append(instance_labels)
					else:
						self.relations[bucket].append(self.labels2id[relations[0]])

					# update attribute value
					self.bucket_values[bucket].append(self.current_value)

					break

	def get_data_buckets(self):

		if self.double_markers:
			return list(self.sentences.values()), list(self.entities_1.values()), list(self.entities_2.values()), list(self.relations.values()), list(self.entities_end1.values()), list(self.entities_end2.values())
		else:
			return list(self.sentences.values()), list(self.entities_1.values()), list(self.entities_2.values()), list(self.relations.values())

	def save_bucket_values(self, file_path):

		with open(file_path, 'a') as file:

			if self.attribute in os.getenv("categorical").split() or self.attribute in os.getenv("aggregate").split():
				for b in self.bucket_values.items():
					file.write(f"{b[0]}\n")
			else:
				for b in self.bucket_values.items():
					if len(b[1]) > 0:
						average = sum(b[1]) / len(b[1])
						file.write(f"{b[0]}\t{str(average).replace('.', ',')}\n")
					else:
						file.write(f"{b[0]}\n")